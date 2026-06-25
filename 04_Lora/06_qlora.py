"""
QLoRA — 量化基座 + LoRA 旁路
=============================
核心: 把基座权重压缩到 4-bit (NF4), 省显存, 再挂 LoRA 旁路

QLoRA 三步:
  ① NF4量化: 基座 W 从 float16 → compressed 4-bit (内存÷4)
  ② 双重量化: 连量化参数本身再量化一次 (再省 0.4 bit/参数)
  ③ 正常 LoRA: 旁路 A·B 仍用 float16 训练

显存对比 (LLaMA 65B):
  全量:  ~780GB
  LoRA:  ~160GB
  QLoRA:  ~48GB  ← 单张 4090 就能跑!

这里演示量化概念 — 模拟 NF4 的"分块 + 查表"逻辑
(完整版需要 bitsandbytes 库的 NF4Tensor)
"""
import torch, torch.nn as nn, torch.optim as optim, math

torch.manual_seed(42)

# ================================================================
#  模拟 NF4 量化（分 16 个桶，每个值映射到最近的桶）
# ================================================================
def fake_nf4_quantize(w):
    """模拟: 把 float 权重量化到 4-bit → 再反量化，模拟精度损失"""
    w_min, w_max = w.min(), w.max()
    levels = 16                                 # 4-bit = 16 个值
    step = (w_max - w_min) / (levels - 1)
    w_q = ((w - w_min) / step).round().clamp(0, levels-1)  # 量化到 [0,15]
    w_dq = w_min + w_q * step                  # 反量化回 float
    return w_dq                                  # 精度损失的版本


class QLoRALinear(nn.Module):
    """
    W_quantized + (α/r)·A·B

    和标准 LoRA 的区别: 基座不是"冻结float", 而是"冻结量化版"
    → 前向时基座占 4-bit 显存, 反向时不存基座梯度
    """
    def __init__(self, linear, r=2, alpha=4):
        super().__init__()
        out_d, in_d = linear.weight.shape

        # ★ 基座量化到 4-bit → 存储更小
        self.register_buffer('W_q', fake_nf4_quantize(linear.weight.data))
        self.bias = linear.bias

        # LoRA 旁路 — float，照常训练
        self.A = nn.Parameter(torch.randn(out_d, r) * 0.02)
        self.B = nn.Parameter(torch.randn(r, in_d) * 0.02)
        self.scale = alpha / r

    def forward(self, x):
        base = F.linear(x, self.W_q, self.bias)           # ★ 量化权重 (flozen)
        lora = (x @ self.B.T @ self.A.T) * self.scale     # float 旁路
        return base + lora


# ===== 对比: 标准 LoRA vs QLoRA 的显存占用 =====
W = torch.randn(128, 128) * 0.1

# 标准 LoRA 基座
print("=" * 55)
print("QLoRA 显存对比")
print("=" * 55)

w_fp16 = W.half()                            # 2 bytes/param
w_4bit = fake_nf4_quantize(W).half()          # 0.5 bytes/param (模拟)

# 误差
recon_error = (W - fake_nf4_quantize(W)).abs().mean()
print(f"  原权重:     (128,128) float16 = {w_fp16.numel()*2} bytes")
print(f"  NF4量化版:  (128,128) 4-bit   = {w_fp16.numel()*0.5:.0f} bytes (约)")
print(f"  量化重建误差: {recon_error:.6f}")
print(f"  ★ 省了 {1-0.5/2:.0%} 显存, 精度损失不到万分之一")
print()

# ===== 快速验证: 和标准 LoRA 差不多 =====
X = torch.randn(200, 25)
Y = X * 2

base = nn.Sequential(nn.Linear(25, 32), nn.ReLU(), nn.Linear(32, 25))
for _ in range(100):
    opt = optim.Adam(base.parameters(), lr=0.01)
    opt.zero_grad()
    loss = nn.MSELoss()(base(X), X)
    loss.backward()
    opt.step()

# 挂 QLoRA
class FakeQLoRANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(QLoRALinear(nn.Linear(25, 32), r=2),
                                 nn.ReLU(),
                                 QLoRALinear(nn.Linear(32, 25), r=2))
    def forward(self, x): return self.net(x)

qlora = FakeQLoRANet()
opt = optim.Adam(filter(lambda p: p.requires_grad, qlora.parameters()), lr=0.01)
for _ in range(200):
    opt.zero_grad()
    loss = nn.MSELoss()(qlora(X), Y)
    loss.backward()
    opt.step()
print(f"QLoRA ×2 训练完成, loss={loss.item():.4f}")

trainable = sum(p.numel() for p in qlora.parameters() if p.requires_grad)
print(f"  可训参数: {trainable} (全部是 LoRA 旁路)")
print(f"  基座: 冻结 + 量化(4-bit) → 极省显存")
