"""
MoCo v1 → v2 → v3 进化史
============================
核心思路：用"动量编码器 + 队列"替代"Memory Bank"

问题链：
  Memory Bank: 存所有样本特征 → 队列满了就踢最旧的
  动量编码器: 用 EMA 更新，保持一致性

v1: 动量编码器 + 队列
v2: 加了 MLP 投影头 + 数据增强
v3: ViT 替代 ResNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

torch.manual_seed(42)

# ===== 数据 =====
N, D = 8, 4
data = torch.randn(N, D)

# ===== 两个编码器 =====
# 查询编码器：正常梯度更新
encoder_q = nn.Linear(D, D)

# 动量编码器：EMA 更新，和 BYOL 的 Target 一样
encoder_k = nn.Linear(D, D)
encoder_k.load_state_dict(encoder_q.state_dict())
for p in encoder_k.parameters():
    p.requires_grad = False

# ===== 队列（替代 Memory Bank）=====
queue_size = 4
queue = deque(maxlen=queue_size)                     # 固定大小，先进先出
for _ in range(queue_size):
    key = F.normalize(encoder_k(torch.randn(1, D)), dim=1)
    queue.append(key.detach())

m = 0.999                                            # 动量系数
temp = 0.07
opt = torch.optim.SGD(encoder_q.parameters(), lr=0.1)

print("=== MoCo: 动量对比 ===")
print(f"队列大小: {queue_size}, 动量 m: {m}\n")

for step in range(5):
    i = torch.randint(0, N, (1,))
    x = data[i]

    # ---- 查询：数据增强 → 编码 ----
    x_q = x + torch.randn(1, D) * 0.05               # 增强1
    q = encoder_q(x_q)
    q = F.normalize(q, dim=1)                         # (1, D)

    # ---- 键：动量编码器（不训）----
    with torch.no_grad():
        x_k = x + torch.randn(1, D) * 0.05           # 增强2
        k_pos = encoder_k(x_k)
        k_pos = F.normalize(k_pos, dim=1)             # (1, D) 正样本键

    # ---- 正负样本得分 ----
    pos_score = (q * k_pos).sum(dim=1) / temp         # (1,)
    neg_keys = torch.cat(list(queue), dim=0)          # (queue_size, D)
    neg_scores = (q @ neg_keys.T).squeeze() / temp    # (queue_size,)

    # ---- InfoNCE ----
    logits = torch.cat([pos_score, neg_scores])       # (1+queue_size,)
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))
    opt.zero_grad()
    loss.backward()
    opt.step()

    # ---- 动量更新 encoder_k ----
    with torch.no_grad():
        for q_p, k_p in zip(encoder_q.parameters(), encoder_k.parameters()):
            k_p.data = m * k_p.data + (1 - m) * q_p.data

    # ---- 当前正样本入队，最旧的自动出队 ----
    queue.append(k_pos.detach())

    print(f"  Step {step}: loss={loss:.3f} | 队列积压={len(queue)}")

print("\n知识点:")
print("  ① Memory Bank → 队列: 固定大小，FIFO，计算量可控")
print("  ② 动量编码器: ξ=m*ξ+(1-m)*θ，缓慢跟随，键编码稳定")
print("  ③ v1→v2: 加了 MLP 投影头 + 更强的数据增强")
print("  ④ v2→v3: ResNet→ViT，骨干网络升级")
print("  ⑤ 队列+动量 = 大batch等效，但不用真的大batch")
