"""
CMC — Contrastive Multiview Coding
=====================================
核心思路：同一个物体的不同"视图"应该编码相近
  视图 = 不同模态（RGB / 深度图 / 文本 / 音频）

和 SimCLR 的区别：
  SimCLR: 同一张图裁两片（同模态）
  CMC:    同一物体的 RGB 图和深度图（跨模态）

这里演示：同一个样本的两种"表示"(x 和 2x+噪声) 应该接近
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据：5个样本，各有两种"视图" =====
N = 5
x_a = torch.randn(N, 4)                             # 视图A: 比如 RGB
x_b = x_a * 2 + torch.randn(N, 4) * 0.1             # 视图B: 比如 深度图（相关但不同）

# ===== 两个编码器（不同模态用不同编码器）=====
encoder_a = nn.Linear(4, 4)                          # RGB 编码器
encoder_b = nn.Linear(4, 4)                          # 深度图编码器
opt = torch.optim.Adam(list(encoder_a.parameters()) + list(encoder_b.parameters()), lr=0.01)

print("=== CMC: 多视图对比 ===\n")

for step in range(20):
    i = torch.randint(0, N, (1,))

    # 编码两个视图
    z_a = F.normalize(encoder_a(x_a[i]), dim=1)      # (1,4) 视图A 编码
    z_b = F.normalize(encoder_b(x_b[i]), dim=1)      # (1,4) 视图B 编码

    # 正样本：同一物体的另一个视图
    pos = (z_a * z_b).sum() / 0.1

    # 负样本：不同物体的视图B
    neg_mask = torch.ones(N, dtype=bool)
    neg_mask[i] = False
    neg_z_b = F.normalize(encoder_b(x_b[neg_mask]), dim=1)
    neg = (z_a @ neg_z_b.T).squeeze() / 0.1

    logits = torch.cat([pos.unsqueeze(0), neg])
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 10 == 0:
        print(f"  Step {step}: pos={pos:.2f} neg_mean={neg.mean():.2f} loss={loss:.3f}")

print("\n知识点:")
print('  ① 跨模态对比：不同"视图"进入不同编码器')
print("  ② 正样本=同一物体的另一种模态, 负样本=其他物体")
print("  ③ 和 SimCLR 核心区别：两个编码器不共享权重（模态不同）")
print("  ④ 训练后：RGB图→Encoder_A 的编码 = 深度图→Encoder_B 的编码")
