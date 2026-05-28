"""
自监督 — MAE 遮罩重建训练方式
================================
任务：输入随机数，遮住一部分，让模型补回来（无标签！）

MAE 核心流程（跟 BERT 一模一样，只是用在连续值上）：
  ① 输入一串数，随机遮掉 75%
  ② Encoder 只看没遮的部分（快的秘诀！）
  ③ Decoder 猜被遮掉的那些位置是什么
  ④ loss = 猜的值 vs 真实值（只在被遮位置算）

和 SimCLR 的区别：
  SimCLR: 正样本拉近、负样本推开（对比）
  MAE:     直接还原被破坏的部分（生成）

和 GAN 的区别：
  GAN:    G生成假图，D判真假（两个网络对抗）
  MAE:    Encoder→Decoder，端到端重建（一个网络还原）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据 =====
N = 300
X = torch.randn(N, 8) * 3                              # (300,8) 每行8个数
Y = X * 2                                                # 真实答案（验证用）


# ===== 模型：Encoder + Decoder =====
encoder = nn.Sequential(
    nn.Linear(8, 32), nn.ReLU(),
    nn.Linear(32, 16),                                   # ★ 只编码可见位置，输出压缩特征
)

decoder = nn.Sequential(
    nn.Linear(16, 32), nn.ReLU(),
    nn.Linear(32, 8),                                    # ★ 重建全部8个位置
)

opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.005)
loss_fn = nn.MSELoss()


# ===== 训练 =====
print("=== MAE: 遮75% → 重建 ===\n")
print(f"{'Epoch':<6} {'loss':<8} {'被遮位还原误差':<15}")

for epoch in range(500):
    x = X[torch.randint(0, N, (64,))]                   # (64, 8)

    # ---- ★ ① 随机遮掉 75% 的位 ----
    mask = torch.rand(64, 8) > 0.75                      # (64,8) True=保留(25%), False=遮掉(75%)
    #       75% 的位置是 False → 被遮

    visible = x * mask.float()                            # (64,8) 被遮位置=0

    # ---- ★ ② Encoder：只看没遮的部分 ----
    feat = encoder(visible)                               # (64,16)

    # ---- ★ ③ Decoder：猜全部位置 ----
    pred = decoder(feat)                                  # (64,8)

    # ---- ★ ④ loss：只算被遮位置的误差 ----
    # mask==False 的位置就是被遮的位置，只有这些位置算loss
    loss = loss_fn(pred[~mask], x[~mask])
    #              ↑ 模型猜的值     ↑ 真实值
    #  只取被遮的那些位置计算：~mask = 取反（遮掉的是False→取反变True→选中）

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 100 == 0:
        # 看看被遮位置的还原效果
        with torch.no_grad():
            mse_masked = F.mse_loss(pred[~mask], x[~mask])
        print(f"{epoch:<6} {loss:.4f}    {mse_masked:.4f}")


# ===== 验证 =====
print(f"\n=== 还原演示 ===")
test_x = X[:3]
mask = torch.rand(3, 8) > 0.75
visible = test_x * mask.float()

with torch.no_grad():
    pred = decoder(encoder(visible))

for i in range(3):
    print(f"\n  原值:    {test_x[i].tolist()}")
    print(f"  遮后:    {[f'{v:.1f}' if m else '?' for v,m in zip(test_x[i], mask[i])]}")
    print(f"  还原:    {[f'{pred[i,j]:.2f}' for j in range(8)]}")


print(f"""
{'='*55}
哪些地方体现了 MAE？
  ① mask = torch.rand(64,8) > 0.75          ← 随机遮75%，MAE的核心
  ② visible = x * mask.float()               ← 遮掉的位变成0
  ③ encoder(visible)                         ← Encoder只处理可见部分(25%)
  ④ pred = decoder(feat)                     ← Decoder还原全部8个位置
  ⑤ loss_fn(pred[~mask], x[~mask])           ← ★ 只在被遮位置算loss！
     ↑ 被遮掉的75%位置才参与损失，没遮的25%不管

对比 SimCLR:
  SimCLR loss = 让同数据的两个增强拉近 (对比)
  MAE loss   = 让被遮的位置猜准原值    (重建)

对比 GAN:
  GAN 有 D判别器和 G生成器两个网络对抗
  MAE 只有一个 Encoder+Decoder 组合，端到端
""")
