"""
SimCLR v1 → v2 进化
=======================
核心思路：最简单的对比学习框架
  同一张图 → 两种随机增强 → 编码 → 投影 → 对比

v1: 大 batch(4096) + 强数据增强 + MLP 投影头
v2: 更大的 MLP 投影头(3层) + 动量编码器(可选)

和 MoCo 的区别：
  MoCo:  小 batch + 队列(存很多负样本)
  SimCLR: 大 batch(负样本全在 batch 里，不用队列)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据：8个样本 =====
N, D = 8, 4
data = torch.randn(N, D)

# ===== v1 编码器 + 投影头 =====
encoder = nn.Linear(D, D)                            # 编码器(ResNet)
# v1: 2层 MLP 投影头
projector = nn.Sequential(                            # (B,D) → (B,4)
    nn.Linear(D, 8), nn.ReLU(),
    nn.Linear(8, 4),                                  # 投影到 4 维
)

# v2 区别：投影头变 3 层
# projector_v2 = nn.Sequential(
#     nn.Linear(D, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
#     nn.Linear(16, 4),
# )

opt = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=0.02)
temp = 0.5

print("=== SimCLR: 简单对比 ===\n")

for step in range(5):
    # ---- 取一个 batch ----
    idx = torch.randint(0, N, (4,))                   # batch_size=4
    x = data[idx]                                      # (4, D)

    # ---- 两种随机增强（模拟）----
    x1 = x + torch.randn(4, D) * 0.2                  # 增强1: 随机裁剪+颜色
    x2 = x + torch.randn(4, D) * 0.2                  # 增强2: 另一种随机

    # ---- 编码 + 投影 ----
    z1 = F.normalize(projector(encoder(x1)), dim=1)   # (4, 4)
    z2 = F.normalize(projector(encoder(x2)), dim=1)   # (4, 4)

    # ---- 构建对比矩阵 ----
    # 把两批拼一起: [z1; z2] (8, 4)
    z = torch.cat([z1, z2], dim=0)                    # (8, 4)
    sim = (z @ z.T) / temp                             # (8, 8) 相似度矩阵
    # sim[i][j] = 第i个和第j个的相似度

    # ---- 正样本标签 ----
    # z1[0] 的正样本是 z2[0]，在拼接后索引是 0 和 4
    # z1[1] 的正样本是 z2[1]，在拼接后索引是 1 和 5
    n = z1.shape[0]                                    # =4
    labels = torch.cat([torch.arange(n) + n, torch.arange(n)])
    # labels = [4,5,6,7, 0,1,2,3] ← 每行的正样本在哪

    # ---- NT-Xent Loss ----
    mask = torch.eye(2*n, dtype=bool)                  # 挖掉对角线（自己和自己的相似度不算）
    sim = sim.masked_fill(mask, float('-inf'))
    loss = F.cross_entropy(sim, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"  Step {step}: loss={loss:.3f} | "
          f"正样本sim均值:{sim[0, 4]:.2f} 负样本sim均值:{sim[0, 1:4].mean():.2f}")

print("\n知识点:")
print("  ① 最简单的框架：Encoder + Projector + 对比Loss")
print("  ② 负样本全来自同 batch（不用 Memory Bank / 队列）")
print("  ③ v1→v2: 投影头从2层→3层，增强更强")
print("  ④ 缺点：需要超大 batch(4096+)，显存吃不消")
print("  ⑤ NT-Xent = 跨视图的交叉熵，每个样本有唯一正样本")
