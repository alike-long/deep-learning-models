"""
自监督 — SimSiam 训练方式
============================
任务：输入随机数，自监督学 ×2（无标签！）

SimSiam 核心：
  ① 同一数据加两次不同噪声 → x1, x2
  ② 共享编码器 + 投影头
  ③ 预测头（SimSiam 独有）去"猜"另一条路径的输出
  ④ stop-gradient：被猜的那条路不传梯度

和 SimCLR 的区别：
  SimCLR:  需要负样本 → 要拼矩阵 → 要算 CrossEntropy → 要大 batch
  SimSiam: 只要正样本 → 同数据两片拉近即可 → stop-gradient 防坍缩

关键一行：
  z2 = projector(encoder(x2)).detach()  ← 切！梯度只从 x1 这条路走
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# ===== 数据 =====
N, D = 200, 1
X = torch.randn(N, D) * 3
Y = X * 2                                 # 真实答案（验证用，不参与训练）


# ===== 模型：共享 Encoder + Projector + Predictor =====
def make_mlp(in_dim, out_dim, hidden=16):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, out_dim))

encoder = make_mlp(1, 8)           # 1→16→8
projector = make_mlp(8, 4)         # 8→16→4
predictor = make_mlp(4, 4)         # ★ SimSiam 独有预测头：4→16→4

opt = torch.optim.SGD(
    list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
    lr=0.05, momentum=0.9
)


def cosine_loss(p, z):
    """余弦距离：0=完全相同，2=完全相反"""
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return (2 - 2 * (p * z).sum(dim=1)).mean()


# ===== 训练 =====
print("=== SimSiam: 自监督学 ×2 ===\n")

for epoch in range(200):
    idx = torch.randint(0, N, (64,))
    x = X[idx]

    # 两种增强：加不同噪声
    x1 = x + torch.randn(64, 1) * 0.1                       # 增强1
    x2 = x + torch.randn(64, 1) * 0.15                      # 增强2

    # ---- 路径1：全过，参与梯度 ----
    z1 = projector(encoder(x1))
    p1 = predictor(z1)

    # ---- 路径2：过半就停 ----
    z2 = projector(encoder(x2)).detach()                     # ★ detach 切断
    #    ^^^^^^^^ encoder 和 projector 在这条路也跑了
    #    但 detach 让梯度停在这里，不从 z2 往回传

    loss = cosine_loss(p1, z2)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 40 == 0:
        with torch.no_grad():
            sim = F.cosine_similarity(F.normalize(p1, dim=1), F.normalize(z2, dim=1)).mean()
        print(f"Epoch {epoch:3d} | loss={loss:.4f} | cos_sim={sim:.3f}")


# ===== 验证 =====
print("\n=== 测试 Encoder ===")
test_x = torch.arange(1, 6).float().view(-1, 1)
with torch.no_grad():
    out = encoder(test_x)
    print(f"  输入 → 编码器输出(8维):")
    for xv, ov in zip(test_x.tolist(), out.tolist()):
        print(f"    f({xv[0]:.0f}) → [{ov[0]:.3f}, {ov[1]:.3f}, ...]")


print(f"""
{'='*55}
知识点:
  ① 没有负样本！只需"同一数据的两种增强编码尽量像"
  ② stop-gradient 阻止坍缩 → 模型不能偷懒输出全0
  ③ Predictor 打破对称 → 两边不一样，避免平凡解
  ④ 和 SimCLR 对比：少了相似度矩阵、少了负样本、少了 CrossEntropy
  ⑤ 训练后 predictor 扔掉，只保留 encoder
""")
