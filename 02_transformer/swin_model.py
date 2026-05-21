"""
Swin Transformer 模型（详细注释版）
===================================
数据流向：
  (B,1,28,28) → Patch切分 → (B,64,4,4)
  → Stage1: SwinBlock×2 (W-MSA + SW-MSA) → (B,4,4,64)
  → PatchMerging: 降采样 → (B,2,2,128)
  → Stage2: 全局注意力 → (B,2,2,128)
  → Pool → (B,128) → Head → (B,10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
#  1. 窗口切分 / 还原（纯函数，没有可学参数）
# ================================================================
def window_partition(x, window_size):
    """
    把一个特征图切成不重叠的小窗口

    比如 x: (2, 8, 8, 64)，window_size=4
      → 切成 4 个 4×4 窗口 → (2*4, 4, 4, 64) = (8, 4, 4, 64)

    为什么要切？
      不做全局注意力（8×8=64个token互相看，成本高）
      只在 4×4=16 个 token 的窗口内做注意力
    """
    B, H, W, C = x.shape
    # 先 reshape：把 H 和 W 轴分别拆成 (H/ws, ws) 和 (W/ws, ws)
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)          # (B, H//ws, ws, W//ws, ws, C)
    # 把"窗口编号"和"窗口内坐标"分开
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()         # (B, H//ws, W//ws, ws, ws, C)
    # 合并 batch 和窗口编号 → 每个窗口变成一个独立样本
    x = x.view(-1, window_size, window_size, C)            # (B*N, ws, ws, C)
    return x


def window_reverse(windows, window_size, H, W):
    """
    window_partition 的逆操作：把窗口拼回完整的特征图
    """
    B_ = windows.shape[0]
    B = B_ // (H // window_size * W // window_size)         # 反推原始 batch
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)       # (B, H//ws, W//ws, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()         # (B, H, W, C)
    x = x.view(B, H, W, -1)
    return x


# ================================================================
#  2. MLP（两层全连接 + GELU）
# ================================================================
class MLP(nn.Module):
    """
    FFN: Linear → GELU → Dropout → Linear → Dropout
    先升维 4 倍再降回来，增加模型容量
    每个 token 独立过，不混合 token 间信息（混合交给 Attention 做）
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * 4)   # (B,N,D) → (B,N,4D)
        self.act = nn.GELU()                          # 平滑激活
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * 4, d_model)   # (B,N,4D) → (B,N,D)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)       # D → 4D
        x = self.act(x)       # GELU
        x = self.drop1(x)
        x = self.fc2(x)       # 4D → D
        x = self.drop2(x)
        return x


# ================================================================
#  3. 窗口多头注意力（Swin 的灵魂）
# ================================================================
class WindowAttention(nn.Module):
    """
    只在窗口内做多头自注意力，而非全局

    QKV 三个矩阵各自独立（不用合并投影），更清晰
    加上"相对位置偏置"让模型知道窗口内每个位置的相对关系
    """
    def __init__(self, d_model=64, n_heads=4, window_size=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads       # 每头的维度，=16
        self.window_size = window_size
        self.scale = self.d_k ** 0.5        # sqrt(d_k)，缩放因子

        # ---- Q, K, V 三个独立投影（不用合并 Linear，更好懂）----
        self.W_q = nn.Linear(d_model, d_model)   # 输入→Query: "我想查什么"
        self.W_k = nn.Linear(d_model, d_model)   # 输入→Key:   "我是什么"
        self.W_v = nn.Linear(d_model, d_model)   # 输入→Value: "我有什么内容"

        # ---- 输出投影 ----
        self.W_o = nn.Linear(d_model, d_model)   # 多头合并后投影

        # ---- 归一层（LayerNorm）+ Dropout ----
        self.norm = nn.LayerNorm(d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_out = nn.Dropout(dropout)

        # ---- 相对位置偏置 ----
        # 窗口内任意两个位置之间有一个可学习的偏置值
        # (2*ws-1)² 种相对位置，每头独立
        num_rel = (2 * window_size - 1) ** 2
        self.rel_bias = nn.Parameter(torch.zeros(num_rel, n_heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(self, x):
        """
        x: (B*W, ws*ws, D)  每个窗口展平成 N=ws*ws 个 token
        比如 ws=4 → N=16, D=64
        → (B*W, N, D)
        """
        B_, N, D = x.shape

        # ---- ① Pre-Norm ----
        shortcut = x
        x = self.norm(x)                              # (B_, N, D)

        # ---- ② 投影 Q, K, V ----
        Q = self.W_q(x)                               # (B_, N, D)
        K = self.W_k(x)                               # (B_, N, D)
        V = self.W_v(x)                               # (B_, N, D)

        # ---- ③ 拆成多头 ----
        Q = Q.view(B_, N, self.n_heads, self.d_k).transpose(1, 2)
        #   (B_, N, D) → (B_, N, h, d_k) → (B_, h, N, d_k)
        K = K.view(B_, N, self.n_heads, self.d_k).transpose(1, 2)
        #   同上
        V = V.view(B_, N, self.n_heads, self.d_k).transpose(1, 2)
        #   同上

        # ---- ④ 注意力得分 = Q @ K^T / sqrt(d_k) ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        #   (B_, h, N, d_k) × (B_, h, d_k, N) → (B_, h, N, N)
        #   N=16, 所以每个窗口内 16×16 的注意力矩阵

        # ---- ⑤ 加相对位置偏置 ----
        rel_bias = self._make_rel_bias(N, scores.device)
        #   (N, N, h) → 广播到 (1, h, N, N)
        scores = scores + rel_bias.permute(2, 0, 1).unsqueeze(0)
        #   每个位置对加一个可学习的偏置

        # ---- ⑥ Softmax + Dropout ----
        attn = F.softmax(scores, dim=-1)             # (B_, h, N, N)  每行权重和=1
        attn = self.drop_attn(attn)

        # ---- ⑦ 加权求和 V ----
        out = torch.matmul(attn, V)                   # (B_, h, N, d_k)
        #   (B_, h, N, N) × (B_, h, N, d_k) → (B_, h, N, d_k)

        # ---- ⑧ 合并多头 ----
        out = out.transpose(1, 2).contiguous()         # (B_, N, h, d_k)
        out = out.view(B_, N, D)                       # (B_, N, D)

        # ---- ⑨ 输出投影 + 残差 ----
        out = self.W_o(out)                           # (B_, N, D)
        out = self.drop_out(out)
        return shortcut + out

    def _make_rel_bias(self, N, device):
        """构建 (N, N, h) 的相对位置偏置矩阵"""
        ws = int(N ** 0.5)                              # window_size
        # 所有位置对的绝对坐标
        coords = torch.arange(ws, device=device)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords = coords.flatten(1)                     # (2, ws*ws) = (2, N)
        # 相对坐标：每对位置的 (dy, dx)
        rel = coords[:, :, None] - coords[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0) + (ws - 1)          # (N, N, 2)，偏移到 [0, 2*ws-2]
        # 把二维坐标压成一维索引
        idx = rel[:, :, 0] * (2 * ws - 1) + rel[:, :, 1]  # (N, N)
        return self.rel_bias[idx]                      # (N, N, h)


# ================================================================
#  4. Swin Transformer 块
# ================================================================
class SwinBlock(nn.Module):
    """
    每个 Block = 窗口注意力 + MLP，都用残差连接

    shift_size = 0: W-MSA  (Window MSA)，窗口正常划分
    shift_size > 0: SW-MSA (Shifted Window MSA)，窗口偏移一半
    """
    def __init__(self, d_model=64, n_heads=4, window_size=4, shift_size=0, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        # ---- 窗口注意力 ----
        self.norm1 = nn.LayerNorm(d_model)              # Pre-Norm
        self.attn = WindowAttention(d_model, n_heads, window_size, dropout)

        # ---- MLP ----
        self.norm2 = nn.LayerNorm(d_model)              # Pre-Norm
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        """
        x: (B, H, W, C)
        → (B, H, W, C)  形状不变
        """
        B, H, W, C = x.shape
        shortcut = x

        # ===== 如果需要移位，先把图滚动 =====
        if self.shift_size > 0:
            # torch.roll: 在 H, W 轴滚动，超出部分循环到另一边
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # 比如 shift=2 时，整张图向左上移 2 格 → 窗口边界全变了
            # 原来不在同一窗口的 patch 现在在一起 → 跨窗通信

        # ===== 窗口注意力 =====
        # ① 切成窗口
        x_windows = window_partition(x, self.window_size)
        #   (B, H, W, C) → (B*Nw, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size ** 2, C)
        #   (B*Nw, ws, ws, C) → (B*Nw, ws*ws, C)  展平窗口内位置

        # ② 注意力 + 残差
        x_windows = self.norm1(x_windows)               # Pre-LN
        x_windows = self.attn(x_windows)                # → (B*Nw, ws*ws, C)

        # ③ 拼回特征图
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x_windows, self.window_size, H, W)
        #   (B*Nw, ws, ws, C) → (B, H, W, C)

        # ===== 如果移位过，滚回来 =====
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # ===== 残差 + MLP =====
        x = shortcut + x                                # 残差1
        x = x + self.mlp(self.norm2(x))                 # 残差2: LN + MLP

        return x


# ================================================================
#  5. Patch Merging（降采样）
# ================================================================
class PatchMerging(nn.Module):
    """
    把 H×W 减半，通道翻倍（类似 CNN 的 stride=2 池化）

    (B, H, W, C) → (B, H/2, W/2, 4C) → LayerNorm → Linear → (B, H/2, W/2, 2C)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim * 4)            # 4 个 patch 拼起来先归一化
        self.reduction = nn.Linear(in_dim * 4, out_dim) # 降维: 4C → 2C

    def forward(self, x):
        """
        x: (B, H, W, C)
        → (B, H/2, W/2, 2C)
        """
        B, H, W, C = x.shape

        # 把相邻 2×2 块的 4 个值抽出来
        x0 = x[:, 0::2, 0::2, :]      # 左上角
        x1 = x[:, 0::2, 1::2, :]      # 右上角
        x2 = x[:, 1::2, 0::2, :]      # 左下角
        x3 = x[:, 1::2, 1::2, :]      # 右下角

        # 拼在通道维 → 4C
        x = torch.cat([x0, x1, x2, x3], dim=-1)          # (B, H/2, W/2, 4C)
        x = self.norm(x)                                 # LayerNorm
        x = self.reduction(x)                            # 4C → 2C
        return x


# ================================================================
#  6. 完整 Swin Transformer
# ================================================================
class SwinTransformer(nn.Module):
    """
    Fashion MNIST 版 Swin（缩小版）

    ┌──────────────────────────────────────────────┐
    │ 输入 (B, 1, 28, 28)                          │
    │   → Conv2d(1→64, kernel=7, stride=7)         │  Patch Embed
    │   → (B, 64, 4, 4)                            │
    │                                               │
    │ Stage1:  SwinBlock×2 (W-MSA + SW-MSA)        │
    │   窗口大小=2, 4×4 分成 2×2=4 个窗口            │
    │   → (B, 4, 4, 64)                            │
    │                                               │
    │ PatchMerging: 降采样 4×4→2×2, 64→128          │
    │   → (B, 2, 2, 128)                            │
    │                                               │
    │ Stage2:  全局注意力（2×2 太小，不用窗口了）       │
    │   → (B, 2, 2, 128)                            │
    │                                               │
    │ Pool + Head:                                  │
    │   → AdaptiveAvgPool2d → (B, 128, 1, 1)        │
    │   → Flatten → (B, 128)                        │
    │   → Linear(128, 10) → (B, 10)                  │
    └──────────────────────────────────────────────┘
    """
    def __init__(self, img_size=28, patch_size=7, num_classes=10,
                 d_model=64, n_heads=4, window_size=2, dropout=0.1):
        super().__init__()

        # ===== ① Patch Embedding =====
        # 用 7×7 卷积把 28×28 图片切成 4×4 个不重叠块
        self.patch_embed = nn.Conv2d(
            in_channels=1,            # 灰度图
            out_channels=d_model,     # 64 通道
            kernel_size=patch_size,   # 7
            stride=patch_size         # 7（步长=卷积核大小→不重叠）
        )
        # 输出: (B, 64, 4, 4)

        # ===== ② Stage1: 窗口注意力 ×2 =====
        self.stage1 = nn.ModuleList([
            # Block 1: W-MSA（窗口不移动）
            SwinBlock(d_model, n_heads, window_size, shift_size=0, dropout=dropout),
            # Block 2: SW-MSA（窗口向右下移 window/2=1 格）
            SwinBlock(d_model, n_heads, window_size, shift_size=window_size // 2, dropout=dropout),
        ])

        # ===== ③ Patch Merging =====
        self.merge1 = PatchMerging(in_dim=d_model, out_dim=d_model * 2)
        # (B, 4, 4, 64) → (B, 2, 2, 128)

        # ===== ④ Stage2: 全局注意力 =====
        # 分辨率降为 2×2=4 个 token，不用窗口了，直接全局注意力
        self.stage2_ln1 = nn.LayerNorm(d_model * 2)     # Pre-Norm
        self.stage2_attn = nn.MultiheadAttention(
            embed_dim=d_model * 2,  # 128
            num_heads=n_heads,      # 4
            dropout=dropout,
            batch_first=True
        )
        self.stage2_ln2 = nn.LayerNorm(d_model * 2)
        self.stage2_mlp = MLP(d_model * 2, dropout)

        # ===== ⑤ 分类头 =====
        self.pool = nn.AdaptiveAvgPool2d(1)             # 空间池化到 1×1
        self.head = nn.Linear(d_model * 2, num_classes) # 128 → 10 类

        # ===== 参数初始化 =====
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """对 Linear 和 LayerNorm 做初始化"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 1, 28, 28) 灰度图
        → (B, 10)          10 类得分
        """
        B = x.shape[0]

        # ===== ① Patch 嵌入 =====
        x = self.patch_embed(x)                         # (B, 1, 28, 28) → (B, 64, 4, 4)
        x = x.permute(0, 2, 3, 1)                      # (B, 4, 4, 64) ← Swin 用 HWC 格式

        # ===== ② Stage1: 窗口注意力 =====
        for block in self.stage1:
            x = block(x)                                # (B, 4, 4, 64) → (B, 4, 4, 64)

        # ===== ③ 降采样 =====
        x = self.merge1(x)                              # (B, 4, 4, 64) → (B, 2, 2, 128)

        # ===== ④ Stage2: 全局注意力 =====
        H, W, C = x.shape[1:]
        x = x.view(B, H * W, C)                         # (B, 4, 128) 展平

        # 子层1: 全局自注意力 + 残差
        shortcut = x
        x = self.stage2_ln1(x)
        attn_out, _ = self.stage2_attn(x, x, x)         # 自注意力: Q=K=V=x
        x = shortcut + attn_out                         # 残差

        # 子层2: MLP + 残差
        x = x + self.stage2_mlp(self.stage2_ln2(x))

        # 恢复空间形状
        x = x.view(B, H, W, C)                          # (B, 2, 2, 128)
        x = x.permute(0, 3, 1, 2)                      # (B, 128, 2, 2) ← 转回 CHW

        # ===== ⑤ 池化 + 分类 =====
        x = self.pool(x)                                # (B, 128, 1, 1)
        x = x.flatten(1)                                # (B, 128)
        x = self.head(x)                                # (B, 10)
        return x
