"""
BYOL — Bootstrap Your Own Latent
====================================
核心思路：不用负样本！两条路自己比

  Online(x) → Projector → Predictor → 预测值
  Target(x') → Projector → 目标值（停梯度）
  让预测值靠近目标值

关键：
  Target 用 EMA 更新（慢速跟随 Online）
  Predictor 打破对称性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据 =====
N, D = 6, 4
data = torch.randn(N, D)


# ===== 辅助函数 =====
def make_mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, out_dim))


# ===== Online 网络 =====
encoder_o = make_mlp(D, D)
projector_o = make_mlp(D, 4)                         # 投影头
predictor = make_mlp(4, 4)                           # ★ BYOL 独有的预测头

# ===== Target 网络（结构相同，EMA 更新）=====
encoder_t = make_mlp(D, D)
projector_t = make_mlp(D, 4)
# 初始时 Target = Online（结构完全一样，只是后续更新方式不同）
encoder_t.load_state_dict(encoder_o.state_dict())
#   ↑ 把 encoder_o 的权重逐层复制给 encoder_t（同结构才能复制）
#   encoder_o.state_dict() → {"0.weight": tensor(...), "0.bias": tensor(...), ...}
#   load_state_dict → encoder_t 的参数全被覆盖为 encoder_o 的当前值
projector_t.load_state_dict(projector_o.state_dict())
#   ↑ 同上，projector 也复制一份

# Target 网络不参与梯度更新，只靠 EMA 缓慢追随 Online
for p in list(encoder_t.parameters()) + list(projector_t.parameters()):
    p.requires_grad = False       # ✂ 切断梯度

opt = torch.optim.Adam(
    list(encoder_o.parameters()) + list(projector_o.parameters()) + list(predictor.parameters()),
    lr=0.01
)
tau = 0.99

print("=== BYOL: 自己引导自己 ===\n")

for step in range(20):
    i = torch.randint(0, N, (1,))
    x = data[i]

    # 两种增强
    x1 = x + torch.randn(1, D) * 0.1
    x2 = x + torch.randn(1, D) * 0.2

    # ---- Online 路径 ----
    z_o = encoder_o(x1)
    z_o = projector_o(z_o)
    pred = predictor(z_o)                            # 预测：Online 想通过这个"猜"Target

    # ---- Target 路径（停梯度）----
    with torch.no_grad():
        z_t = encoder_t(x2)
        z_t = projector_t(z_t)                       # 目标：Target 的编码（不动）

    # ---- 损失：让预测接近目标 ----
    pred = F.normalize(pred, dim=1)
    z_t = F.normalize(z_t, dim=1)
    loss = 2 - 2 * (pred * z_t).sum()               # 余弦距离 = 2-2*cos_sim

    opt.zero_grad()
    loss.backward()
    opt.step()

    # ---- EMA 更新 Target ----
    with torch.no_grad():
        for o_p, t_p in zip(encoder_o.parameters(), encoder_t.parameters()):
            t_p.data = tau * t_p.data + (1 - tau) * o_p.data
        for o_p, t_p in zip(projector_o.parameters(), projector_t.parameters()):
            t_p.data = tau * t_p.data + (1 - tau) * o_p.data

    if step % 10 == 0:
        print(f"  Step {step:2d}: loss={loss:.4f} | cos_sim={(pred*z_t).sum():.3f}")

print("\n知识点:")
print("  ① ★ 不需要负样本！对比学习也可以不对比")
print("  ② Online → Predictor: 多一层，打破对称→避免坍缩")
print("  ③ Target 用 EMA 更新：慢速追随，提供稳定目标")
print("  ④ 为什么不坍缩？Predictor + EMA 共同阻止")
print("  ⑤ 比 SimCLR 简单，效果差不多，且不依赖大 batch")
