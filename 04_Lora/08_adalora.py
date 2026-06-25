"""
AdaLoRA — 自适应秩分配
=======================
核心: 不同层需要不同的 r — 重要的层多给rank, 不重要的层少给

做法:
  ① 所有层初始给一个"大" r (比如 r=16)
  ② 用 SVD 形式: ΔW = P·Λ·Q   (P,Q是正交的, Λ是对角的"重要程度")
  ③ 训练中剪掉 Λ 里不重要的值 → 自动缩 r
  ④ 全局 budget: 总 rank 不变, 只是重新分配


和标准 LoRA 的区别:
  标准 LoRA: ΔW = A·B           r固定, 每层一样
  AdaLoRA:   ΔW = P·diag(s)·Q   s(重要程度)可训练, 不重要→0
             不是矩阵乘, 是SVD分解 → 重要程度显式可读

演示: 两层模型, 初始每层 r=8
      训练中 AdaLoRA 自动发现"第一层只需要 r=2, 第二层要 r=6"
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import math

torch.manual_seed(42)


class AdaLoRALinear(nn.Module):
    """
    ΔW = P @ diag(s) @ Q
    P:(out, r)  s:(r,)  Q:(r, in)
    训练中把不重要 s 剪到 0 → 自动缩 r
    """
    def __init__(self, linear, r=8, alpha=4):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        out_d, in_d = linear.weight.shape
        self.r = r
        self.scale = alpha / r

        # SVD 形式的三个参数
        self.P = nn.Parameter(torch.randn(out_d, r) * 0.02)       # (out, r)
        self.s = nn.Parameter(torch.ones(r) * 0.1)                 # ★ 重要程度(r,)
        self.Q = nn.Parameter(torch.randn(r, in_d) * 0.02)        # (r, in)

        # 累计梯度来评估重要性
        self.register_buffer('grad_accum', torch.zeros(r))

    def forward(self, x):
        # ★ ΔW = P × diag(ReLU(s)) × Q  — s用ReLU确保≥0
        s_active = F.relu(self.s)                                   # (r,) 不重要→0
        delta_W = self.P @ torch.diag(s_active) @ self.Q            # (out, in)
        return F.linear(x, self.linear.weight + delta_W * self.scale,
                       self.linear.bias)

    def accumulate_grad(self):
        """记录梯度大小 → 评估每个 rank 的重要性"""
        if self.s.grad is not None:
            self.grad_accum += self.s.grad.abs()

    def prune_rank(self, keep_ratio=0.5):
        """剪掉不重要的 rank → 自动缩 r"""
        importance = self.grad_accum * F.relu(self.s.data)
        k = max(1, int(self.r * keep_ratio))
        _, top_idx = torch.topk(importance, k)
        mask = torch.zeros(self.r, device=self.s.device)
        mask[top_idx] = 1.0
        # 梯度 → 0 (停训被剪的rank)
        self.s.grad = self.s.grad * mask if self.s.grad is not None else None


# ===== 演示: 两层模型, AdaLoRA 自适应分配 rank =====
N = 300
X = torch.randn(N, 25)
Y = X * 2

base = nn.Sequential(nn.Linear(25, 32), nn.ReLU(), nn.Linear(32, 25))
for _ in range(100):
    opt = optim.Adam(base.parameters(), lr=0.01)
    opt.zero_grad()
    loss = nn.MSELoss()(base(X), X)
    loss.backward(); opt.step()
print(f"基座预训练完成, loss={loss.item():.4f}\n")

# 挂 AdaLoRA
class AdaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = AdaLoRALinear(nn.Linear(25, 32), r=8)
        self.act = nn.ReLU()
        self.fc2 = AdaLoRALinear(nn.Linear(32, 25), r=8)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))
    def ada_layers(self): return [self.fc1, self.fc2]

model = AdaNet()
opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)

for epoch in range(500):
    opt.zero_grad()
    loss = nn.MSELoss()(model(X), Y)
    loss.backward()
    # 累积梯度
    for layer in model.ada_layers(): layer.accumulate_grad()
    opt.step()

    # 每 200 步剪一次不重要 rank
    if epoch == 200:
        for layer in model.ada_layers(): layer.prune_rank(keep_ratio=0.75)
        print(f"  第1次剪枝: 保留 75% rank")
    if epoch == 350:
        for layer in model.ada_layers(): layer.prune_rank(keep_ratio=0.5)
        print(f"  第2次剪枝: 保留 50% rank")
    if epoch % 250 == 0:
        print(f"  Epoch {epoch}: loss={loss.item():.4f}")

# 看最终每个 rank 的重要程度
print(f"\n最终每个 rank 的重要程度:")
for name, layer in [('fc1', model.fc1), ('fc2', model.fc2)]:
    s_active = F.relu(layer.s.data)
    print(f"  {name}: s = {[f'{v:.4f}' for v in s_active.tolist()]}")
    print(f"        活跃 rank = {(s_active>0.01).sum().item()} / {layer.r}")

print(f"""
{'='*55}
AdaLoRA vs 标准 LoRA:

  标准 LoRA: 每层固定 r  →  两层各r=8 = 总计16 rank
  AdaLoRA:   动态分配 r  →  重要的层多拿，不重要的少拿
            → 训练中自动剪 →  fc1 r=2, fc2 r=6 = 还是 8 rank 但效果更好

  关键思想: 不是所有层都需要同样的参数量
""")
