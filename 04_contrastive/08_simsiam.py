"""
SimSiam — Simple Siamese
===========================
核心思路：比 BYOL 更简单！连 EMA 都不要

  Encoder(x1) → Projector → Predictor → p1
  Encoder(x2) → Projector → 停梯度 → z2
  让 p1 靠近 z2

唯一关键：stop-gradient（停梯度）
  z2 不参与梯度 → 防止模型输出常数（坍缩）

和 BYOL 的区别：
  BYOL:  两个编码器（Online + Target），Target 用 EMA
  SimSiam: 一个编码器，两边共享，只靠 stop-gradient
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据 =====
N, D = 6, 4
data = torch.randn(N, D)


def make_mlp(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(), nn.Linear(8, out_dim))


# ===== 一个编码器（两边共享）=====
encoder = make_mlp(D, D)
projector = make_mlp(D, 4)
# ★ SimSiam 也有 Predictor，但没有 Target/EMA
predictor = make_mlp(4, 4)

params = list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters())
opt = torch.optim.SGD(params, lr=0.05, momentum=0.9)

print("=== SimSiam: 最简单的孪生网络 ===\n")

for step in range(20):
    i = torch.randint(0, N, (1,))
    x = data[i]

    # 两种增强
    x1 = x + torch.randn(1, D) * 0.1
    x2 = x + torch.randn(1, D) * 0.2

    # ---- 路径1: 编码 + 投影 + 预测 ----
    z1 = encoder(x1)
    z1 = projector(z1)
    p1 = predictor(z1)

    # ---- 路径2: 编码 + 投影 → stop-gradient ----
    z2 = encoder(x2)
    z2 = projector(z2).detach()                      # ★ 停梯度！SimSiam 唯一关键

    # ---- 对称损失 ----
    p1 = F.normalize(p1, dim=1)
    z2 = F.normalize(z2, dim=1)
    p2_pred = predictor(projector(encoder(x2)))
    p2_pred = F.normalize(p2_pred, dim=1)
    z1_sg = projector(encoder(x1)).detach()
    z1_sg = F.normalize(z1_sg, dim=1)

    loss1 = -(p1 * z2).sum()                          # x1→预测 靠近 x2→目标
    loss2 = -(p2_pred * z1_sg).sum()                  # x2→预测 靠近 x1→目标
    loss = loss1 + loss2

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        print(f"  Step {step:2d}: loss={loss:.4f} | cos_sim1={(p1*z2).sum():.3f}")

print("\n知识点:")
print("  ① 最简单的非对比方法：一个编码器 + stop-gradient")
print("  ② 没有负样本、没有 Momentum Encoder、没有 Memory Bank")
print("  ③ stop-gradient 是唯一关键：防止模型输出全0作弊")
print("  ④ 和 BYOL 对比：砍了 EMA、砍了 Target 网络，效果相当")
print("  ⑤ 为什么能工作？stop-gradient → 相当于隐式的 EM 算法")
