"""
InstDisc — 实例判别 + Memory Bank
====================================
核心思路：每张图就是一个类别（N 张图 = N 类）
用 Memory Bank 存所有样本的特征，省去每次重新算

对比目标：自己的增强版本 = 正样本，Memory Bank 里其他所有图 = 负样本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据：5个"图片"，每图2维 =====
N, D = 5, 2                                      # 5 个样本，每个 2 维
data = torch.randn(N, D)                           # 假装是 5 张图的特征
data = F.normalize(data, dim=1)                    # 归一化到单位球面

# ===== Memory Bank: 存所有样本的特征 =====
memory = data.clone().detach()                     # (5, 2) — 每个样本存一个特征
memory = F.normalize(memory, dim=1)

# ===== 模型：简单编码器 =====
encoder = nn.Linear(D, D)
opt = torch.optim.SGD(encoder.parameters(), lr=0.1)

temp = 0.07                                        # 温度：越小越"硬"
print("=== InstDisc: 实例判别 ===")
print(f"Memory Bank 存了 {N} 个样本的特征，每轮更新\n")

for step in range(3):
    i = torch.randint(0, N, (1,))                  # 随机选一张"图"
    x = data[i]                                     # 原图特征
    x_aug = x + torch.randn(1, D) * 0.05           # 数据增强版本

    # ---- 编码 ----
    q = encoder(x)                                  # (1, 2)  查询
    q = F.normalize(q, dim=1)

    # ---- 正样本得分 ----
    pos = encoder(x_aug)                            # 增强版本
    pos = F.normalize(pos, dim=1)
    pos_score = (q * pos).sum() / temp              # 点积 / 温度

    # ---- 负样本得分：Memory Bank 里所有其他样本 ----
    neg_mask = torch.ones(N, dtype=bool)
    neg_mask[i] = False                             # 挖掉自己
    neg = memory[neg_mask]                          # (4, 2)
    neg_scores = (q @ neg.T) / temp                 # (1, 4)

    # ---- NCE Loss ----
    logits = torch.cat([pos_score.view(1), neg_scores.view(-1)])
    label = torch.tensor([0])                       # 第0个是正样本
    loss = F.cross_entropy(logits.view(1, -1), label)

    opt.zero_grad()
    loss.backward()
    opt.step()

    # ---- 更新 Memory Bank: 把自己这项换掉 ----
    with torch.no_grad():
        memory[i] = encoder(data[i]).detach()
        memory = F.normalize(memory, dim=1)

    print(f"  Step {step}: loss={loss:.3f} | "
          f"正样本得分:{pos_score:.2f} | 负样本均值:{neg_scores.mean():.2f}")

print("\n知识点:")
print("  ① 每张图 = 一个类别 → N张图有N类")
print("  ② Memory Bank 缓存所有特征，避免每轮前向N次")
print("  ③ 自己的增强版本 = 正，Memory Bank 里其他所有 = 负")
print("  ④ 每次更新后，把自己新的特征写回 Memory Bank 对应位置")
