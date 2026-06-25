"""
DoRA — 权重分解 + LoRA (Weight-Decomposed Low-Rank Adaptation)
=================================================================
核心: 把权重分解成"幅度 × 方向", LoRA 只管方向, 幅度单独学

标准 LoRA:   W' = W + A·B            ← 幅度+方向混在一起改
DoRA:        W' = m × (W + A·B)/norm  ← 幅度(m)和方向分开

拆开的好处:
  幅度 m: 决定"这个特征有多重要" (标量 × C_out)
  方向:   W + A·B (归一化后) — LoRA 只管这里

为什么比原始 LoRA 好?
  1. 幅度和方向分别学 → 更稳定
  2. 初始化 m = ||W|| → 初始时完全等价于原始模型
  3. 微调时幅度动得慢，方向动得快 → 不会把原始知识冲掉
"""
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

torch.manual_seed(42)


class DoRALinear(nn.Module):
    """
    W' = m × (W + A·B) / ||W + A·B||
         ↑标量   ↑方向 (单位向量)
    """
    def __init__(self, linear, r=2, alpha=4):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False                   # 冻基座 W

        out_d, in_d = linear.weight.shape
        self.r = r
        self.scale = alpha / r

        # LoRA 旁路 — 只管方向
        self.A = nn.Parameter(torch.randn(out_d, r) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, in_d))  # 从 0 开始

        # ★ 幅度向量 — 初始化为原始权重的各通道范数
        with torch.no_grad():
            init_m = linear.weight.norm(p=2, dim=1)               # (out_d,)
        self.m = nn.Parameter(init_m)                              # (out_d,)

    def forward(self, x):
        # ★ 新权重 = 幅度 × 归一化方向
        delta_W = self.A @ self.B                                 # (out_d, in_d) LoRA增量
        W_new = self.linear.weight + delta_W * self.scale         # 原始+LoRA
        W_dir = F.normalize(W_new, p=2, dim=1)                    # ★ 归一化到单位向量
        W_dora = self.m.unsqueeze(1) * W_dir                      # ★ 幅度×方向

        return F.linear(x, W_dora, self.linear.bias)


# ===== 对比: 标准 LoRA vs DoRA =====
X = torch.randn(200, 25)
Y = X * 2

class LoRABase(nn.Module):
    """标准 LoRA"""
    def __init__(self):
        super().__init__()
        out_d, in_d = 25, 25
        self.W = nn.Parameter(torch.randn(out_d, in_d)*0.1, requires_grad=False)
        self.A = nn.Parameter(torch.randn(out_d,2)*0.02)
        self.B = nn.Parameter(torch.zeros(2,in_d))
    def forward(self, x):
        return F.linear(x, self.W + self.A@self.B*2.0)

class DoRABase(nn.Module):
    """DoRA"""
    def __init__(self):
        super().__init__()
        out_d, in_d = 25, 25
        self.W = nn.Parameter(torch.randn(out_d,in_d)*0.1, requires_grad=False)
        self.A = nn.Parameter(torch.randn(out_d,2)*0.02)
        self.B = nn.Parameter(torch.zeros(2,in_d))
        self.m = nn.Parameter(self.W.norm(p=2,dim=1).clone())
    def forward(self, x):
        W_new = self.W + self.A@self.B*2.0
        W_dir = F.normalize(W_new, p=2, dim=1)
        return F.linear(x, self.m.unsqueeze(1)*W_dir)

# 训练两个模型
lora_model = LoRABase()
dora_model = DoRABase()
opt_lora = optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=0.01)
opt_dora = optim.Adam(filter(lambda p: p.requires_grad, dora_model.parameters()), lr=0.01)

for _ in range(300):
    opt_lora.zero_grad(); opt_dora.zero_grad()
    loss_l = nn.MSELoss()(lora_model(X), Y)
    loss_d = nn.MSELoss()(dora_model(X), Y)
    loss_l.backward(); loss_d.backward()
    opt_lora.step(); opt_dora.step()

print(f"标准 LoRA loss={loss_l.item():.5f} | DoRA loss={loss_d.item():.5f}")

# 看 DoRA 的幅度学到什么
lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
dora_params = sum(p.numel() for p in dora_model.parameters() if p.requires_grad)
print(f"LoRA可训: {lora_params} | DoRA可训: {dora_params} (多了幅度向量 {dora_model.m.numel()})")

print(f"""
{'='*55}
DoRA vs LoRA 核心区别:
  LoRA:  W' = W + A·B           (增量和原始混一起)
  DoRA:  W' = m × (W+A·B)/norm  (幅度和方向分开)

  幅度 m: 决定"这个feature channel有多重要"
  方向 W_dir: 单位向量 = 特征本身的形状
  → 微调时改变特征方向, 但幅度跟原始权重一致 → 更稳定
""")
