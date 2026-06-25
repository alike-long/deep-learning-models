"""
ViT — Vision Transformer
==========================
图片 → 切成Patch → 展平 → 加位置编码 → Transformer Encoder → 分类

和 NLP Transformer 一样，只是"词 → patch (图块)"
和 Swin 的关键区别见文件末尾
"""
import torch
import torch.nn as nn


# ================================================================
#  ViT 模型 — 预处理是核心
# ================================================================
class ViT(nn.Module):
    """
    输入 (B, 3, 224, 224) → 分类 (B, num_classes)
    """
    def __init__(self, num_classes=10, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.d_model = d_model
        self.patch_size = 16                         # 16×16 小块
        self.n_patches = (224 // 16) ** 2            # 14×14 = 196 个 patch
        self.patch_dim = 3 * 16 * 16                 # 每 patch 3×16×16 = 768 像素

        # ---- ① Patch 嵌入：768 像素 → 128维 ----
        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        #   (B, 196, 768) → (B, 196, 128)

        # ---- ② [CLS] token：放在 196 个 patch 前面，代表"整张图" ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        #   形状 (1, 1, 128) → 广播到 (B, 1, 128)

        # ---- ③ 位置编码：196 个 patch + 1 个 [CLS] = 197 个位置 ----
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        #   形状 (1, 197, 128)

        # ---- ④ Transformer Encoder ----
        self.enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.enc_layer, n_layers)

        # ---- ⑤ 分类头：只取 [CLS] 位置的输出 ----
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        完整的维度变化 (以 B=2 为例):
          x: (2, 3, 224, 224)      ← 原始 RGB 图片
          → patches: (2, 196, 768) ← 切成 14×14=196 个小块
          → embedded: (2, 196, 128) ← 每块投影到 128 维
          → + cls_token: (2, 197, 128) ← 最前面加 [CLS]
          → + pos_embed: (2, 197, 128) ← 加位置编码
          → Transformer ×3: (2, 197, 128) ← 197 个 token 全局互看
          → cls_out: (2, 128)        ← 取 [CLS] 位置
          → logits: (2, 10)          ← 分类
        """
        B = x.shape[0]

        # ===== ① 图片 → Patch =====
        # (B,3,224,224) → (B,3,14,16,14,16) → (B,14,14,3,16,16)
        x = x.unfold(2, self.patch_size, self.patch_size) \
             .unfold(3, self.patch_size, self.patch_size)
        #    unfold：沿着 H 和 W 轴滑窗取块，步长=块大小（不重叠）

        # (B,14,14,3,16,16) → (B,14,14,768) → (B,196,768)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        #   把 C 通道调到 H,W 后面：→ (B,14,14,3,16,16)
        #   ↑ 这步不改变数据，只是调维序
        x = x.view(B, self.n_patches, -1)
        #   把 14×14 展平 → 196，把 3×16×16 展平 → 768

        # ===== ② 投影到 d_model =====
        x = self.patch_embed(x)                          # (B, 196, 128)

        # ===== ③ 前面加 [CLS] token =====
        cls = self.cls_token.expand(B, -1, -1)           # (1, 1, 128) → (B, 1, 128)
        x = torch.cat([cls, x], dim=1)                   # (B, 197, 128)
        #   197 个 token：[CLS] + patch0 + patch1 + ... + patch195

        # ===== ④ 加位置编码 =====
        x = x + self.pos_embed                           # (B, 197, 128)
        #   每个位置有独特的 128 维位置信号

        # ===== ⑤ Transformer Encoder（全局注意力！）=====
        x = self.encoder(x)                              # (B, 197, 128)
        #   197 个 token 互相看：197×197 的注意力矩阵 = 38,809 对关系

        # ===== ⑥ 取 [CLS] → 分类 =====
        x = self.norm(x[:, 0])                           # (B, 128) 取第 0 列
        return self.head(x)                              # (B, 10)


if __name__ == "__main__":
    print("=== ViT 维度演示 ===\n")
    model = ViT(num_classes=10)
    x = torch.randn(2, 3, 224, 224)                     # 2 张假 RGB 图

    with torch.no_grad():
        out = model(x)

    print(f"输入:     {tuple(x.shape)}     ← 2张 224×224 RGB图")
    print(f"                                            ")
    print(f"  ① unfold 切块:                              ")
    print(f"     (2,3,224,224) → unfold(H) → (2,3,14,16,14,16)")
    print(f"                   → unfold(W) → (2,3,14,16,14,16)")
    print(f"                   → permute  → (2,14,14,3,16,16)")
    print(f"                   → view     → (2, 196, 768)  ← 14×14=196个, 3×16²=768像素/个")
    print(f"                                            ")
    print(f"  ② patch_embed: (2,196,768) → (2, 196, 128) ← 投影到 d_model")
    print(f"  ③ +cls_token:  → (2, 197, 128) ← 前面加[CLS]")
    print(f"  ④ +pos_embed:  → (2, 197, 128) ← 加位置编码")
    print(f"  ⑤ Transformer: → (2, 197, 128) ← 3层全局注意力")
    print(f"  ⑥ cls_out:     → (2, 128)      ← 只取[CLS]")
    print(f"  ⑦ head:        → (2, 10)       ← 分类")
    print(f"                                            ")
    print(f"输出:     {tuple(out.shape)}")
    print(f"\n{'='*60}")
    print(f"ViT vs Swin Transformer 对比")
    print(f"{'='*60}")
    print(f"""
  ┌────────────┬─────────────────┬─────────────────────┐
  │            │ ViT             │ Swin                │
  ├────────────┼─────────────────┼─────────────────────┤
  │ 注意力     │ 全局            │ 窗口内              │
  │ 复杂度     │ O(N²)          │ O(W² × N/W²)        │
  │ 分辨率     │ 始终不变        │ 逐层降采样          │
  │ 感受野     │ 第一层就全局    │ 浅层局部，深层全局  │
  │ 位置编码   │ 可学/不可学     │ 相对位置偏置        │
  │ 像谁       │ NLP Transformer │ CNN (分层+局部)  │
  │ 大图效率   │ 差 (196²≈38k)  │ 好 (窗内算)        │
  │ 小图/      │ 可以            │ 更好 (多尺度)      │
  │ 分类 /     │ 强 (数据够时)   │ 强 (数据少也能训)  │
  │ 检测 /     │ 弱              │ 强 (多尺度天生适合) │
  └────────────┴─────────────────┴─────────────────────┘

  核心差异：ViT = 图片当作文本（全局注意力），Swin = 图片当分层信号（局部→全局）
""")
