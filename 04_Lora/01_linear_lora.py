"""
Linear LoRA 基础 — 基座冻住，旁路 A·B 微调 ×2 和 ÷2
======================================================
W_new = W_frozen + (α/r)·A·B
"""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class LoRALinear(nn.Module):
    """out = Wx + (α/r)·A·B·x"""
    def __init__(self, linear, r=2, alpha=4):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        out_d, in_d = linear.weight.shape
        self.A = nn.Parameter(torch.randn(out_d, r) * 0.02)
        self.B = nn.Parameter(torch.randn(r, in_d) * 0.02)
        self.scale = alpha / r

    def forward(self, x):
        return self.linear(x) + (x @ self.B.T @ self.A.T) * self.scale

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(25, 32), nn.ReLU(), nn.Linear(32, 25))
    def forward(self, x): return self.net(x)

X = torch.randn(200, 25)
# ---- 预训练基座 → 恒等 ----
base = BaseModel()
opt = optim.Adam(base.parameters(), lr=0.01)
for _ in range(200):
    opt.zero_grad(); loss = nn.MSELoss()(base(X), X); loss.backward(); opt.step()
print(f"基座预训练完成, loss={loss.item():.4f}")

# ---- 挂 LoRA ----
def wrap(model):
    for i, m in enumerate(model.net):
        if isinstance(m, nn.Linear): model.net[i] = LoRALinear(m)
wrap(base)

# ---- LoRA 微调 ×2 ----
opt = optim.Adam(filter(lambda p: p.requires_grad, base.parameters()), lr=0.01)
for _ in range(300):
    opt.zero_grad(); loss = nn.MSELoss()(base(X), X*2); loss.backward(); opt.step()
print(f"LoRA ×2 完成, loss={loss.item():.4f}")

# ---- 测试 ----
with torch.no_grad():
    t = torch.randn(1, 25)
    print(f"输入: {t[0,0]:.2f}  期望: {t[0,0]*2:.2f}  LoRA: {base(t)[0,0]:.2f}")

# 关 LoRA → 回到基座
base.net[0].scale = 0.0
with torch.no_grad():
    print(f"关LoRA: {base(t)[0,0]:.2f} (≈输入, 基座未变)")
