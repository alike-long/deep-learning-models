"""
SwAV — Swapping Assignments between Views
============================================
核心思路：不直接对比两个视图，而是对比它们的"聚类分配"

流程：
  同一张图 → 两种增强 → 编码 → 投影
  z1 → 预测聚类 Softmax → 伪标签 q1
  z2 → 预测聚类 Softmax → 伪标签 q2
  让 z1 预测 q2，z2 预测 q1（交换！）

好处：不需要两两对比，复杂度从 O(N²) 降到 O(NK)
      K = 聚类数(通常3000)，远小于 N
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据 =====
N, D = 8, 4
data = torch.randn(N, D)

# ===== 编码器 + 投影头 + 原型向量 =====
K = 3                                                # 聚类数 = 3 个原型
encoder = nn.Linear(D, D)
projector = nn.Linear(D, K)                          # → K 维 = K 个聚类的得分
prototypes = nn.Parameter(torch.randn(K, D) * 0.1)   # (K, D) 可学习的原型向量

opt = torch.optim.Adam(
    list(encoder.parameters()) + list(projector.parameters()) + [prototypes],
    lr=0.02
)

print("=== SwAV: 交换聚类分配 ===\n")

# Sinkhorn 简化为 Softmax（Sinkhorn 只是加了等分约束）
def get_assignment(z, prototypes):
    """把一个视图的特征分配给聚类"""
    scores = z @ prototypes.T / 0.1                  # (B, K) 相似度
    return F.softmax(scores * 10, dim=1)             # 硬一点的 Softmax → 伪标签

for step in range(10):
    i = torch.randint(0, N, (4,))
    x = data[i]

    # 两种增强
    x1 = x + torch.randn(4, D) * 0.2
    x2 = x + torch.randn(4, D) * 0.2

    # 编码 → 投影
    z1 = F.normalize(encoder(x1), dim=1)             # (4, D)
    z2 = F.normalize(encoder(x2), dim=1)

    # 计算两种视图的"聚类分配"
    q1 = get_assignment(z1, prototypes.detach())     # (4, K) 视图1的伪标签
    q2 = get_assignment(z2, prototypes.detach())     # (4, K) 视图2的伪标签

    # ★ 交叉预测：用 z1 预测 q2，用 z2 预测 q1
    p1 = F.softmax(z1 @ prototypes / 0.1, dim=1)     # (4, K) z1预测的聚类
    p2 = F.softmax(z2 @ prototypes / 0.1, dim=1)     # (4, K) z2预测的聚类

    # 损失：交换！让预测去匹配另一个视图的伪标签
    loss = -(q2 * torch.log(p1 + 1e-8)).sum() - (q1 * torch.log(p2 + 1e-8)).sum()
    loss = loss / 4                                   # 平均

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 5 == 0:
        print(f"  Step {step}: loss={loss:.3f} | "
              f"q1 argmax={q1.argmax(dim=1).tolist()} | q2={q2.argmax(dim=1).tolist()}")

print("\n知识点:")
print("  ① 不直接比两个视图的特征 → 比它们的聚类分配")
print("  ② 交叉预测: 用视图1预测视图2的聚类 → 交换")
print("  ③ Sinkhorn: 让-Batch 内聚类分配均匀（每个聚类分到的样本差不多）")
print("  ④ O(N²) → O(NK): N个样本 vs N个样本 → N个样本 vs K个聚类")
print("  ⑤ 原型向量 = 可学习的聚类中心")
