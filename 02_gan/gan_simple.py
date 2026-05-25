"""
DCGAN — 生成 Fashion MNIST 图片
=================================
生成器: 随机噪声(100维) → 反卷积 → (1, 28, 28) 假图
判别器: (1, 28, 28) 图片 → 卷积 → 0(假)/1(真)

训练目标:
  D: 真图判1，假图判0
  G: 生成假图，让 D 判为 1（骗过 D）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ================================================================
#  1. 生成器 — 噪声 → 图片
# ================================================================
class Generator(nn.Module):
    """
    输入: (B, 100, 1, 1)  随机噪声
    输出: (B, 1, 28, 28)  伪造的灰度图

    ConvTranspose2d = 反卷积：把低分辨率"放大"成高分辨率
      (1,1) → (7,7) → (14,14) → (28,28)
    """
    def __init__(self, noise_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            # Block1: (B,100,1,1) → (B,256,7,7)
            nn.ConvTranspose2d(noise_dim, 256, 7, 1, 0),
            #   in=100 out=256 kernel=7 stride=1 padding=0
            #   out_size = (1-1)*1 + 7 - 0 = 7
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Block2: (B,256,7,7) → (B,128,14,14)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            #   out_size = (7-1)*2 + 4 - 2*1 = 14
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Block3: (B,128,14,14) → (B,1,28,28)
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            #   out_size = (14-1)*2 + 4 - 2*1 = 28
            nn.Tanh(),          # 压到 [-1, 1]，和真实图归一化一致
        )

    def forward(self, z):
        # z: (B, 100, 1, 1)  ← 噪声向量加重塑
        return self.net(z)    # (B, 1, 28, 28)


# ================================================================
#  2. 判别器 — 图片 → 真/假
# ================================================================
class Discriminator(nn.Module):
    """
    输入: (B, 1, 28, 28)  可能是真图或假图
    输出: (B, 1)          0~1

    Conv2d：从图片逐层压缩，最后输出一个判断值
    (28,28) → (14,14) → (7,7) → (1,1) → 得分
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block1: (B,1,28,28) → (B,64,14,14)
            nn.Conv2d(1, 64, 4, 2, 1),
            #   out_size = (28 - 4 + 2*1)/2 + 1 = 14
            nn.LeakyReLU(0.2),          # GAN 用 LeakyReLU 防神经元死

            # Block2: (B,64,14,14) → (B,128,7,7)
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # Block3: (B,128,7,7) → (B,256,3,3)
            nn.Conv2d(128, 256, 3, 2, 1),
            #   out_size = (7 - 3 + 2*1)/2 + 1 = 3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 最后：自适应池化 → 展开 → 判定
            nn.AdaptiveAvgPool2d(1),     # (B,256,3,3) → (B,256,1,1)
            nn.Flatten(),                 # → (B, 256)
            nn.Linear(256, 1),            # → (B, 1)
            nn.Sigmoid(),                 # → 0~1
        )

    def forward(self, x):
        return self.net(x)               # (B, 1)


# ================================================================
#  3. 训练
# ================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    noise_dim = 100

    # ---- 真数据：Fashion MNIST，归一化到 [-1, 1] ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))   # [0,1] → [-1,1]，和 Tanh 对齐
    ])
    dataset = datasets.FashionMNIST("../01_data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # ---- 模型 ----
    G = Generator(noise_dim).to(device)
    D = Discriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    #                                ↑ DCGAN 标配参数
    loss_fn = nn.BCELoss()

    print(f"Fashion MNIST DCGAN | 设备: {device}\n")

    for epoch in range(30):
        total_D, total_G = 0, 0
        for real_img, _ in loader:
            B = real_img.size(0)
            real_img = real_img.to(device)

            # ===== ① 训 D: 真→1, 假→0 =====
            # 真图
            real_pred = D(real_img)                     # (B, 1)
            loss_D_real = loss_fn(real_pred, torch.ones(B, 1, device=device))

            # 假图
            z = torch.randn(B, noise_dim, 1, 1, device=device)
            fake_img = G(z).detach()                    # detach: 只训 D
            fake_pred = D(fake_img)
            loss_D_fake = loss_fn(fake_pred, torch.zeros(B, 1, device=device))

            loss_D = loss_D_real + loss_D_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ===== ② 训 G: 假图→让 D 判为 1 =====
            z = torch.randn(B, noise_dim, 1, 1, device=device)
            fake_img = G(z)                              # 不 detach，训 G
            fake_pred = D(fake_img)
            loss_G = loss_fn(fake_pred, torch.ones(B, 1, device=device))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            total_D += loss_D.item()
            total_G += loss_G.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | D:{total_D/len(loader):.4f} | G:{total_G/len(loader):.4f}")

    print("\n训练完成！")
    return G


# ================================================================
#  4. 生成图片 + 保存
# ================================================================
def generate_and_save(G, noise_dim=100, num=16, save_path="gan_fashion.png"):
    """生成 num 张假 Fashion MNIST 图并保存"""
    G.eval()
    device = next(G.parameters()).device

    with torch.no_grad():
        z = torch.randn(num, noise_dim, 1, 1, device=device)
        fake = G(z).cpu()                           # (16, 1, 28, 28)

    # 画 4×4 网格
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake[i, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle("GAN 生成的 Fashion MNIST", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"图片已保存到 {save_path}")


# ================================================================
#  5. 运行
# ================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    G = train()
    generate_and_save(G)
