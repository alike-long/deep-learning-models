"""
CPC — Contrastive Predictive Coding
=====================================
核心思路：用过去预测未来
  不是把一个图切两片比（SimCLR 方式）
  而是：给前几个时间步 → 预测未来的特征

自回归模型 → 上下文向量 → 预测未来 → 和真实未来比
正样本=真实的未来，负样本=随机的其他位置
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据：一条"时间序列"，10个时间步，每步4维 =====
T, D = 10, 4
seq = torch.randn(T, D)                              # 10 步时间序列

# ===== 自回归模型（GRU）：用过去预测上下文 =====
gru = nn.GRU(D, 8, batch_first=True)                 # → 8维隐藏态
# 线性预测头：每一步预测后面几步的特征
pred_head = nn.Linear(8, D)

params = list(gru.parameters()) + list(pred_head.parameters())
opt = torch.optim.Adam(params, lr=0.01)

print("=== CPC: 对比预测编码 ===\n")

# 输入：前 6 步 → 预测第 7 步
context = seq[:6].unsqueeze(0)                       # (1, 6, 4)
_, h = gru(context)                                  # h: (1, 1, 8)
c = h.squeeze(0).squeeze(0)                          # (8,) 上下文向量

# 预测第 7 步的特征
pred = pred_head(c)                                  # (4,)
target = seq[6]                                      # (4,) 真实的未来

# 正样本: 真实未来
pos = F.cosine_similarity(pred, target, dim=0)

# 负样本: 序列里的其他时间步
neg_idx = torch.tensor([0, 2, 4, 8])                # 随机不相关的时间步
neg_targets = seq[neg_idx]                            # (4, 4)
neg = F.cosine_similarity(pred.unsqueeze(0), neg_targets, dim=1)

# InfoNCE Loss
logits = torch.cat([pos.unsqueeze(0), neg])
label = torch.tensor([0])
loss = F.cross_entropy(logits.unsqueeze(0) * 2, label)  # ×2放大温度

print(f"  正样本(步7): {pos:.4f}  负样本均值: {neg.mean():.4f}  loss:{loss:.4f}")

print("\n知识点:")
print("  ① 不是空间对比(裁两片)，是时间对比(前→后)")
print("  ② 自回归(GRU/LSTM) → 上下文向量 → 预测未来特征")
print("  ③ 正样本 = 真实的未来，负样本 = 随机其他时间步")
print("  ④ 学到的上下文向量包含了对未来有用的信息")
