"""
最简 GAN — 输入 3×3 矩阵，学会乘 2
=====================================

玩法：
  生成器: 拿 3×3 矩阵，试图"猜"它的 ×2 版本
  判别器: 看到 3×3 矩阵，判断是"真正的 ×2" 还是"生成器伪造的 ×2"

训练到最后：
  输入 [[1,2,3],[4,5,6],[7,8,9]]
  生成器输出 ≈ [[2,4,6],[8,10,12],[14,16,18]]

GAN 精髓：生成器想骗过判别器，判别器想识破生成器，双方对抗中一起进步
"""
import torch
import torch.nn as nn
import torch.optim as optim


# ================================================================
#  1. 生成器（Generator）
# ================================================================
class Generator(nn.Module):
    """
    输入:  (B, 9)  3×3 展平
    输出:  (B, 9)  伪造的 ×2 版本

    就像一个假钞工厂：拿到真钞样式，模仿出假钞
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 16),      # (B, 9) → (B, 16)
            nn.ReLU(),
            nn.Linear(16, 16),     # (B, 16) → (B, 16)
            nn.ReLU(),
            nn.Linear(16, 9),      # (B, 16) → (B, 9)  ← 输出也是 9 个数
        )

    def forward(self, x):
        # x: (B, 9)  ← 原始 3×3 展平
        return self.net(x)  # (B, 9)  ← 试图输出 ×2 的值


# ================================================================
#  2. 判别器（Discriminator）
# ================================================================
class Discriminator(nn.Module):
    """
    输入:  (B, 9)  一个 3×3 展平（可能是真 ×2，也可能是假的）
    输出:  (B, 1)  0~1，0=假，1=真

    就像一个验钞员：拿到纸币，判断是真钞还是假钞
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 16),      # (B, 9) → (B, 16)
            nn.ReLU(),
            nn.Linear(16, 8),      # (B, 16) → (B, 8)
            nn.ReLU(),
            nn.Linear(8, 1),       # (B, 8) → (B, 1)
            nn.Sigmoid(),          # 压到 0~1
        )

    def forward(self, x):
        # x: (B, 9)
        return self.net(x)  # (B, 1)


# ================================================================
#  3. 造数据集
# ================================================================
def make_batch(batch_size=64):
    """生成一批数据：随机 3×3 + 真正的 ×2 版本"""
    # 随机 3×3（值在 0~10 之间）
    x_real = torch.rand(batch_size, 9) * 10    # (B, 9)    原始矩阵
    y_real = x_real * 2                         # (B, 9)    真实的 ×2

    return x_real, y_real


# ================================================================
#  4. 训练
# ================================================================
def train():
    G = Generator()
    D = Discriminator()

    opt_G = optim.Adam(G.parameters(), lr=0.001)    # 生成器优化器
    opt_D = optim.Adam(D.parameters(), lr=0.001)    # 判别器优化器
    loss_fn = nn.BCELoss()                           # 二分类交叉熵

    print("开始训练 GAN...")
    for epoch in range(3000):
        # ---- 每一轮制造一批新数据 ----
        x_real, y_real = make_batch(batch_size=128)
        #   x_real: (128, 9)  原始 3×3
        #   y_real: (128, 9)  真 ×2

        # ===== ① 训练判别器 =====
        # 真样本：判别器应该判为 1
        real_pred = D(y_real)                              # (128, 1)
        loss_D_real = loss_fn(real_pred, torch.ones(128, 1))
        #                                    ↑ 真样本标签=1

        # 假样本：生成器伪造的 ×2，判别器应该判为 0
        fake = G(x_real).detach()                          # (128, 9)
        #        ↑ .detach(): 不让梯度流回生成器，只训判别器
        fake_pred = D(fake)                                # (128, 1)
        loss_D_fake = loss_fn(fake_pred, torch.zeros(128, 1))
        #                                    ↑ 假样本标签=0

        loss_D = loss_D_real + loss_D_fake
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ===== ② 训练生成器 =====
        # 生成器目标：骗过判别器，让它判为 1
        fake = G(x_real)                                    # (128, 9)
        fake_pred = D(fake)                                 # (128, 1)
        loss_G = loss_fn(fake_pred, torch.ones(128, 1))
        #                              ↑ 假的也要骗判别器说=1

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # ---- 日志 ----
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | "
                  f"D_loss:{loss_D.item():.4f} | "
                  f"G_loss:{loss_G.item():.4f}")

    print("训练完成！\n")
    return G


# ================================================================
#  5. 测试
# ================================================================
def test(G):
    """输入一个 3×3 矩阵，看生成器能不能输出 ×2 的结果"""
    G.eval()

    # 测试矩阵
    test_input = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]])

    with torch.no_grad():
        flat = test_input.view(1, 9)          # (1, 9) 展平
        output = G(flat).view(3, 3)            # (3, 3) 恢复形状

    print("=" * 40)
    print("测试结果")
    print("=" * 40)
    print(f"输入矩阵:\n{test_input}\n")
    print(f"期望 (×2):\n{test_input * 2}\n")
    print(f"生成器输出:\n{output}\n")

    # 算一下误差
    target = test_input * 2
    error = torch.abs(output - target)
    print(f"绝对误差:\n{error}\n")
    print(f"平均误差: {error.mean().item():.4f}")

    # ===== 你也可以自己输入 =====
    print("\n" + "=" * 40)
    print("手动测试（输入 -1 退出）")
    while True:
        try:
            s = input("\n输入 9 个数字（空格分隔）: ").strip()
            if s == "-1":
                break
            nums = [float(x) for x in s.split()]
            if len(nums) != 9:
                print("要输入 9 个数！")
                continue
            inp = torch.tensor(nums).view(1, 9)
            with torch.no_grad():
                out = G(inp).view(3, 3)
            print(f"生成器输出:\n{out}")
        except (ValueError, EOFError, KeyboardInterrupt):
            break


# ================================================================
#  6. 运行
# ================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    G = train()
    test(G)
