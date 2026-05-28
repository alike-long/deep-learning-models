"""
MAE 预训练 → 骨架给 GAN 生成器 → 生成 Fashion MNIST 图片
===========================================================
分两步：
  Step1: MAE 预训练（遮75%图块→Encoder→Decoder→还原被遮块）
         学会"图像结构"，Encoder 变成好骨架
  Step2: 把 MAE Encoder 当 GAN 生成器的特征提取器
         GAN 用骨架参数，不再从头训

MAE 怎么处理图片？
  28×28 → 切 4×4=16 个 patch (每块 7×7=49 像素)
  → 随机遮 75%(12块) → Encoder 只看 4 块 → Decoder 还原 16 块
  → 只在被遮的 12 块上算 loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math, random
import matplotlib.pyplot as plt

torch.manual_seed(42)

# ===== 设备 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")

# ===== 超参数 =====
IMG_SIZE = 28              # Fashion MNIST 图片尺寸
PATCH_SZ = 7               # 每块 7×7 → 4×4=16 块
N_PATCHES = (IMG_SIZE//PATCH_SZ)**2   # 16
PATCH_DIM = PATCH_SZ*PATCH_SZ        # 每块 49 个像素
MASK_RATIO = 0.75          # 遮 75% = 留 4 块
D_MODEL = 64               # Transformer 维度
BATCH = 64


# ================================================================
#  工具函数：图片 ↔ Patch 互转
# ================================================================
def img_to_patches(x):
    """
    把图片切成不重叠的小块，每块展平成向量
    x: (B, 1, 28, 28) → (B, N=16, 49)
    就像把一张图切成 4×4=16 块拼图，每块是 49 个像素
    """
    B, C, H, W = x.shape
    # 切块: (B,1,28,28) → (B,1,4,7,4,7) → (B,4,4,7,7)
    x = x.view(B, C, H//PATCH_SZ, PATCH_SZ, W//PATCH_SZ, PATCH_SZ)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    # (B,4,4,1,7,7) → (B,4,4,49) → (B,16,49)
    x = x.view(B, N_PATCHES, -1)
    return x


def patches_to_img(x):
    """
    把 patch 拼回图片（img_to_patches 的逆操作）
    x: (B, 16, 49) → (B, 1, 28, 28)
    """
    B = x.shape[0]
    x = x.view(B, IMG_SIZE//PATCH_SZ, IMG_SIZE//PATCH_SZ, 1, PATCH_SZ, PATCH_SZ)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, 1, IMG_SIZE, IMG_SIZE)
    return x


# ================================================================
#  MAE Encoder — 只看没被遮的块（ViT 风格）
# ================================================================
class MAEEncoder(nn.Module):
    """
    1. 把每个 49 维 patch 投影到 64 维
    2. 加位置编码（不然 transformer 不知道哪些块是邻居）
    3. 过 N 层 Transformer Encoder
    4. 输出：被遮位置的可学习 token + 可见块的编码拼接
    """
    def __init__(self):
        super().__init__()
        # ① 把每个 patch 从 49 维像素投影到 64 维特征
        self.patch_proj = nn.Linear(PATCH_DIM, D_MODEL)
        # ② 位置编码：16 个位置各 64 维
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, D_MODEL) * 0.02)
        # ③ [MASK] token：所有被遮位置共用一个可学习向量
        self.mask_token = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        # ④ Transformer 编码器层
        self.enc_layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.enc_layer, num_layers=2)

    def forward(self, patches, mask):
        """
        patches: (B, 16, 49)  原始 patch
        mask:    (B, 16)      True=保留(可见), False=遮掉
        返回 visible 块 + mask 块拼接后的编码
        """
        B = patches.shape[0]

        # ---- ① 只投影可见块！MAE 加速的秘诀 ----
        x = self.patch_proj(patches)                    # (B, 16, 64)

        # ---- ② 拼：可见块 + [MASK] token ----
        # 可见位置用真实 patch 编码，遮掉位置用 mask_token 占位
        mask_expand = mask.unsqueeze(-1).float()         # (B, 16, 1)
        x = x * mask_expand + self.mask_token * (1 - mask_expand)
        #   可见位=patch编码    被遮位=mask_token (可学习)

        # ---- ③ 加位置编码 ----
        x = x + self.pos_embed                           # (B, 16, 64)

        # ---- ④ Transformer 编码 ----
        x = self.encoder(x)                              # (B, 16, 64)
        return x


# ================================================================
#  MAE Decoder — 拿编码还原被遮的块
# ================================================================
class MAEDecoder(nn.Module):
    """
    输入：Encoder 的 16 个编码（有些是真实块，有些是 mask_token 编码）
    输出：16 个重建的 patch（每个 49 像素）
    """
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, D_MODEL) * 0.02)
        self.dec_layer = nn.TransformerEncoderLayer(
            D_MODEL, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(self.dec_layer, num_layers=2)
        # 最后投影：64 维 → 49 像素（一个 patch）
        self.head = nn.Linear(D_MODEL, PATCH_DIM)

    def forward(self, x):
        x = x + self.pos_embed                           # (B, 16, 64)
        x = self.decoder(x)                              # (B, 16, 64)
        x = self.head(x)                                 # (B, 16, 49)  ← 每个位置还原 49 个像素
        return x


# ================================================================
#  GAN Generator — 噪声 → 图片（骨架来自 MAE Encoder）
# ================================================================
class GANGenerator(nn.Module):
    """
    1. 随机噪声 → Linear → 变成 16 个 patch token → Decoder → 假图
    2. 用的是 MAE 预训练好的 Decoder + 位置编码
    3. 噪声通过投影变成 patch 级特征，然后用 Decoder 还原成图
    """
    def __init__(self, mae_encoder, mae_decoder):
        super().__init__()
        self.mae_enc = mae_encoder                            # ★ 预训练好的 Encoder
        self.mae_dec = mae_decoder                            # ★ 预训练好的 Decoder
        # 噪声投影：100维噪声 → 16个 patch 的 64维特征
        self.noise_to_patches = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, N_PATCHES * D_MODEL),              # → 1024 = 16*64
        )

    def forward(self, z):
        B = z.shape[0]
        # ① 噪声 → 16 个 patch token
        fake_patches = self.noise_to_patches(z)                # (B, 1024)
        fake_patches = fake_patches.view(B, N_PATCHES, D_MODEL) # (B, 16, 64)
        # ② Decoder 还原成像素
        fake_patches = self.mae_dec(fake_patches)              # (B, 16, 49)
        # ③ patch → 图片
        img = patches_to_img(fake_patches)                      # (B, 1, 28, 28)
        return img.to(z.device)                                  # ★ 确保和噪声同一设备


# ================================================================
#  GAN Discriminator — 判真图还是假图
# ================================================================
class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),    # (B,1,28,28) → (B,32,14,14)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),   # → (B,64,7,7)
            nn.LeakyReLU(0.2),
            nn.Flatten(),                  # → (B, 64*7*7)
            nn.Linear(64*7*7, 1),          # → (B, 1)
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
#  Step1: MAE 预训练
# ================================================================
def train_mae(loader):
    print("=" * 55)
    print("Step1: MAE 预训练 — 遮75%图块 → 还原")
    print("=" * 55)

    encoder = MAEEncoder().to(device)
    decoder = MAEDecoder().to(device)
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    for epoch in range(5):
        total_loss = 0
        for imgs, _ in loader:
            imgs = imgs[:BATCH].to(device)                    # (B,1,28,28)

            # ---- ① 切 patch + 随机遮 75% ----
            patches = img_to_patches(imgs)                     # (B, 16, 49)
            B = patches.shape[0]

            # 生成 mask：每行 16 个位置，随机选 4 个保留（25%）
            mask = torch.zeros(B, N_PATCHES, dtype=bool, device=device)
            for b in range(B):
                idx = torch.randperm(N_PATCHES, device=device)[: int(N_PATCHES * (1 - MASK_RATIO))]
                mask[b, idx] = True                            # True=保留

            # ---- ② Encoder → Decoder ----
            enc_out = encoder(patches, mask)                   # (B, 16, 64)
            pred = decoder(enc_out)                            # (B, 16, 49)

            # ---- ③ ★ loss 只在被遮位置算 ----
            loss_mask = ~mask                                  # 取反 → 被遮的位
            loss = F.mse_loss(pred[loss_mask], patches[loss_mask])

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1} | 被遮块复原loss: {total_loss/len(loader):.4f}")

    print("  MAE 预训练完成！Encoder 学会了图像结构\n")
    return encoder, decoder


# ================================================================
#  Step2: GAN 训练 — MAE 骨架 + GAN 头
# ================================================================
def train_gan(loader, mae_enc, mae_dec):
    print("=" * 55)
    print("Step2: GAN 训练 — MAE骨架 → 生成器")
    print("=" * 55)

    G = GANGenerator(mae_enc, mae_dec).to(device)
    D = GANDiscriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    for epoch in range(10):
        total_d, total_g = 0, 0
        for imgs, _ in loader:
            imgs = imgs[:BATCH].to(device)
            B = imgs.shape[0]

            # ---- 训 D: 真图→1, 假图→0 ----
            real_pred = D(imgs)
            loss_d_real = loss_fn(real_pred, torch.ones(B, 1, device=device))

            z = torch.randn(B, 100, device=device)
            fake_imgs = G(z).detach()                          # detach：只训D
            fake_pred = D(fake_imgs)
            loss_d_fake = loss_fn(fake_pred, torch.zeros(B, 1, device=device))

            loss_d = loss_d_real + loss_d_fake
            opt_D.zero_grad(); loss_d.backward(); opt_D.step()

            # ---- 训 G: 假图→骗D判1 ----
            z = torch.randn(B, 100, device=device)
            fake_imgs = G(z)                                   # 不detach：训G
            loss_g = loss_fn(D(fake_imgs), torch.ones(B, 1, device=device))

            opt_G.zero_grad(); loss_g.backward(); opt_G.step()
            total_d += loss_d.item()
            total_g += loss_g.item()

        print(f"  Epoch {epoch+1:2d} | D:{total_d/len(loader):.4f} | G:{total_g/len(loader):.4f}")
    print("  GAN 训练完成！\n")
    return G


# ================================================================
#  可视化：生成图 vs 原图
# ================================================================
def compare(G, loader):
    """画 2 行：上行=原图（Fashion MNIST），下行=生成图"""
    G.eval()
    real_imgs = next(iter(loader))[0][:8].to(device).cpu()    # (8,1,28,28)

    with torch.no_grad():
        fake_imgs = G(torch.randn(8, 100, device=device)).cpu()  # (8,1,28,28)

    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
        axes[0, i].imshow(real_imgs[i, 0], cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("原图", fontsize=10)

        axes[1, i].imshow(fake_imgs[i, 0], cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("MAE+GAN生成", fontsize=10)

    plt.suptitle("原图 vs MAE预训练骨架 + GAN 生成的图", fontsize=12)
    plt.tight_layout()
    plt.savefig("mae_gan_result.png", dpi=100)
    plt.show()
    print("对比图已保存: mae_gan_result.png")


# ================================================================
#  运行
# ================================================================
if __name__ == "__main__":
    print("MAE预训练 → GAN生成 完整流程\n")

    # 数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))                  # → [-1,1]
    ])
    dataset = datasets.FashionMNIST("01_data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    # Step1: MAE 预训练
    mae_enc, mae_dec = train_mae(loader)

    # Step2: GAN 用 MAE 骨架生成图片
    G = train_gan(loader, mae_enc, mae_dec)

    # 对比原图 vs 生成图
    compare(G, loader)
