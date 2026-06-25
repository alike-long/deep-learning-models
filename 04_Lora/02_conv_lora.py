"""
卷积 LoRA — 基座恒等 → LoRA旁路(1×1×2) → 边缘检测
=====================================================
每步都标了维度变化，重点关注 forward 里各层的进出形状
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据: (B=300, C=1, H=4, W=4) =====
N = 300
X = torch.randn(N, 1, 4, 4) * 0.5              # (300, 1, 4, 4)

# Sobel 边缘算出的目标
sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]], device='cpu').view(1,1,3,3)
sobel_y = torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]], device='cpu').view(1,1,3,3)
Y = F.conv2d(F.pad(X, (1,1,1,1)), sobel_x).abs() + \
    F.conv2d(F.pad(X, (1,1,1,1)), sobel_y).abs()  # (300, 1, 4, 4)


# ================================================================
#  ConvLoRA 详细维度
# ================================================================
class ConvLoRA(nn.Module):
    """
    ┌───────────── 维度一览 (以 B=300, C=1, H=W=4 为例) ─────────────┐
    │                                                                 │
    │  输入 x: (B, 1, 4, 4)                                           │
    │                                                                 │
    │  === 基座路径 (冻结) ===                                         │
    │  self.conv = Conv2d(1→1, k=3, pad=1)  权重: (1,1,3,3) = 9参数  │
    │    → base: (B, 1, 4, 4)            ← H和W不变(padding=1保尺寸)   │
    │                                                                 │
    │  === LoRA 路径 (可训) ===                                        │
    │  self.down = Conv2d(1→2, k=1)       权重: (2,1,1,1) = 2参数     │
    │    → down_out: (B, 2, 4, 4)         ← H,W不变, 通道 1→2(r)      │
    │  self.up   = Conv2d(2→1, k=1)       权重: (1,2,1,1) = 2参数     │
    │    → up_out:   (B, 1, 4, 4)         ← H,W不变, 通道 2→1         │
    │                                                                 │
    │  === 合并 ===                                                    │
    │  scale = α/r = 4/2 = 2                                          │
    │  output = base + lora×scale         ← 两路相加, 形状不变        │
    │         = (B,1,4,4) + (B,1,4,4)                                │
    │         = (B, 1, 4, 4)                                           │
    └─────────────────────────────────────────────────────────────────┘
    """
    def __init__(self, conv, r=2, alpha=4):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False

        in_c  = conv.in_channels                       # 1
        out_c = conv.out_channels                      # 1
        self.scale = alpha / r                         # 2.0

        # ★ 两个 1×1 卷积 = 逐点 LoRA 旁路
        self.down = nn.Conv2d(in_c, r, 1, bias=False) # (1→2, k=1) 权重(2,1,1,1)
        self.up   = nn.Conv2d(r, out_c, 1, bias=False)# (2→1, k=1) 权重(1,2,1,1)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        """
        x:       (B, 1, 4, 4)
        base:    (B, 1, 4, 4)    ← Conv2d(1→1,k=3) 冻结
        down_out:(B, 2, 4, 4)    ← Conv2d(1→2,k=1) channel先缩→r
        up_out:  (B, 1, 4, 4)    ← Conv2d(2→1,k=1) channel再扩→out
        output:  (B, 1, 4, 4)    ← base + up_out*scale
        """
        base = self.conv(x)                            # (B, 1, 4, 4)
        lora = self.up(self.down(x))                   # (B, 1, 4, 4)
        return base + lora * self.scale                # (B, 1, 4, 4)


# ================================================================
#  Step1: 预训练基座 → 恒等平滑
# ================================================================
conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
#   权重: (out_c=1, in_c=1, kh=3, kw=3) = 9 个参数

opt = optim.Adam(conv.parameters(), lr=0.01)
for _ in range(100):
    opt.zero_grad()
    loss = nn.MSELoss()(conv(X), X)
    loss.backward()
    opt.step()
print(f"基座预训练完成 | 恒等 loss={loss.item():.5f}")
print(f"  基座权重 (1,1,3,3): {[f'{v:.3f}' for v in conv.weight.data.flatten()]}")

# ================================================================
#  Step2: LoRA 微调 → 边缘检测
# ================================================================
lora = ConvLoRA(conv, r=2, alpha=4)
#   self.conv.weight: (1,1,3,3) = 9参数 ← 冻结
#   self.down.weight: (2,1,1,1) = 2参数 ← 可训
#   self.up.weight:   (1,2,1,1) = 2参数 ← 可训
#   总计: 9冻 + 4训 = 13参数

print(f"\n前向传播维度 (B=300):")
print(f"  x            → (300, 1, 4, 4)")
print(f"  self.conv(x) → (300, 1, 4, 4)  冻 Conv2d(1→1,k=3)")
print(f"  self.down(x) → (300, 2, 4, 4)  训 Conv2d(1→2,k=1) ↓通道压缩")
print(f"  self.up(...) → (300, 1, 4, 4)  训 Conv2d(2→1,k=1) ↑通道还原")
print(f"  output       → (300, 1, 4, 4)  base + up(down)*2")
print()

opt = optim.Adam(filter(lambda p: p.requires_grad, lora.parameters()), lr=0.005)
for epoch in range(300):
    opt.zero_grad()
    out = lora(X)                                       # (300,1,4,4)
    loss = nn.MSELoss()(out, Y)
    loss.backward()
    opt.step()
    if epoch % 150 == 0:
        print(f"  微调 epoch {epoch:3d} | loss={loss.item():.5f}")

print(f"\n微调完成！")

# ================================================================
#  验证
# ================================================================
with torch.no_grad():
    test = torch.randn(3, 1, 4, 4)                      # (3, 1, 4, 4)
    base_out = conv(test)                                # (3, 1, 4, 4)
    lora_out = lora(test)                                # (3, 1, 4, 4)
    edge = F.conv2d(F.pad(test, (1,1,1,1)), sobel_x).abs() + \
           F.conv2d(F.pad(test, (1,1,1,1)), sobel_y).abs()

print(f"\n验证 (第1个样本前4个像素):")
print(f"  基座 (≈恒等):     {[f'{v:.3f}' for v in base_out[0,0,0]]}")
print(f"  LoRA (≈边缘检测): {[f'{v:.3f}' for v in lora_out[0,0,0]]}")
print(f"  目标 (真实Sobel): {[f'{v:.3f}' for v in edge[0,0,0]]}")

# ================================================================
#  和 Linear LoRA 对比维度
# ================================================================
base_n = sum(p.numel() for p in conv.parameters())
lora_n = sum(p.numel() for p in lora.parameters() if p.requires_grad)
print(f"\n参数量: 基座 {base_n} 冻 + LoRA {lora_n} 训 = {base_n+lora_n} 总")

print(f"""
{'='*55}
维度对比: Linear LoRA vs Conv LoRA
================================

  Linear:  x(B, in) → W_frozen·x  +  (α/r)·(A·B·x)
           A:(out, r)  B:(r, in)  ← 两个矩阵直接乘

  Conv:    x(B,C,H,W) → Conv3×3(x) + (α/r)·Conv1×1_up(Conv1×1_down(x))
           down:(r, C, 1,1)  ← 每个像素独立过 C→r 线性变换
           up:  (out, r, 1,1) ← 每个像素独立过 r→out 线性变换
           1×1 卷积 = kernel_size=1 → 不跨像素混合 → 逐点 Linear

  为什么 1×1 是"逐点 Linear"?
    Conv2d(C→R, k=1): 对每个(H,W)位置, 做一次 Linear(R = x@W + b)
    无空间混合, 只做通道变换 → 等价于对每个像素的C维向量做矩阵乘!
""")
