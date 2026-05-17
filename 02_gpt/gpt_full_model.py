"""
GPT 完整模型 — 手写每一步，带维度变化
======================================
用途：根据开头往下续写文本

核心公式：
  输入 x: (B, T)            ← id序列
  Embedding: (B, T, D)      ← 词嵌入 + 位置嵌入
  N层 Block: (B, T, D) → (B, T, D)  ← 每层不变形状
  输出头: (B, T, D) → (B, T, V)     ← 每个位置预测下一个字

B = batch, T = seq_len, D = d_model, V = vocab_size

和 BERT 关键区别：
  BERT = Encoder, 双向看, 做理解
  GPT  = Decoder, 单向看(因果), 做生成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ================================================================
#  1. 因果自注意力（GPT 的灵魂）
# ================================================================
class CausalSelfAttention(nn.Module):
    """
    因果 = 只能看当前位置及左边，不能偷看未来

    步骤：
      ① 输入 x 分别过 Wq/Wk/Wv  →  Q, K, V
      ② 拆成多头: (B,T,D) → (B,h,T,D//h)
      ③ QK^T / sqrt(d_k)  →  得分矩阵 (B,h,T,T)
      ④ 加因果mask：上三角=-inf  →  softmax后权重=0
      ⑤ softmax → 权重 × V  →  输出
      ⑥ 合并多头 → W_o
    """
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads                    # 每个头的维度

        # Q, K, V, O 四个投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        """
        x: (B, T, D)  →  输出: (B, T, D)
        causal_mask: (T, T) 上三角=-inf
        """
        B, T, D = x.shape

        # ---- ① 线性投影 ----
        Q = self.W_q(x)   # (B, T, D)  → (B, T, D)
        K = self.W_k(x)   # (B, T, D)  → (B, T, D)
        V = self.W_v(x)   # (B, T, D)  → (B, T, D)

        # ---- ② 拆多头 ----
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        #   (B, T, D) → (B, T, h, d_k) → (B, h, T, d_k)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        #   (B, T, D) → (B, T, h, d_k) → (B, h, T, d_k)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        #   (B, T, D) → (B, T, h, d_k) → (B, h, T, d_k)

        # ---- ③ 注意力得分 ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #   (B, h, T, d_k) × (B, h, d_k, T) → (B, h, T, T)
        #   ^除sqrt(d_k) 防点积过大→softmax梯度消失

        # ---- ④ 因果mask ----
        if causal_mask is not None:
            # causal_mask: (T, T)，上三角=-inf
            scores = scores + causal_mask
            # -inf 位置经过 softmax → 0，彻底看不见

        # ---- ⑤ softmax + 加权 ----
        attn = F.softmax(scores, dim=-1)          # (B, h, T, T) 每一行权重和=1
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)                # (B, h, T, d_k)

        # ---- ⑥ 合并多头 ----
        out = out.transpose(1, 2).contiguous()     # (B, T, h, d_k)
        out = out.view(B, T, D)                    # (B, T, D)
        out = self.W_o(out)                        # (B, T, D)
        return out


# ================================================================
#  2. 前馈网络（FFN / MLP）
# ================================================================
class FeedForward(nn.Module):
    """
    Linear → GELU → Linear
    先升维（D → 4D）再降回来（4D → D），增加模型容量
    GPT 用 GELU 不是 ReLU（GELU 更平滑，梯度不截断）
    """
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),   # (B, T, D) → (B, T, 4D)
            nn.GELU(),                           # 激活（不改变形状）
            nn.Linear(d_model * 4, d_model),    # (B, T, 4D) → (B, T, D)
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)   # (B,T,D) → (B,T,D)


# ================================================================
#  3. GPT Block（一个 Transformer 层）
# ================================================================
class GPTBlock(nn.Module):
    """
    每层做两件事：
      ① 因果自注意力 → 理解"上下文"
      ② FFN → 非线性变换，增加表达能力
    每次操作后：残差连接 + LayerNorm

    这里用 Pre-Norm（先Norm再做操作，最后残差加回去）
    GPT-2/3 都用 Pre-Norm，训练更稳定
    """
    def __init__(self, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)           # 注意力前的 Norm
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)           # FFN 前的 Norm
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, causal_mask=None):
        # ---- 子层1：因果自注意力 ----
        x = x + self.attn(self.ln1(x), causal_mask)
        #   ln1(x): (B,T,D) → (B,T,D)
        #   attn(): (B,T,D) → (B,T,D)
        #   +x残差: (B,T,D) → (B,T,D)

        # ---- 子层2：FFN ----
        x = x + self.ffn(self.ln2(x))
        #   ln2(x): (B,T,D) → (B,T,D)
        #   ffn():  (B,T,D) → (B,T,D)
        #   +x残差: (B,T,D) → (B,T,D)

        return x   # 形状始终 (B, T, D)


# ================================================================
#  4. GPT 完整模型
# ================================================================
class GPT(nn.Module):
    """
    完整流程:
      输入 id: (B, T)
        → Token Embed (B, T, D) × sqrt(D)
        → + Position Embed (B, T, D)
        → Dropout
        → N 层 GPTBlock (B, T, D) → ... → (B, T, D)
        → LayerNorm (B, T, D)
        → Linear(D→V) (B, T, V)  ← 每个位置预测词表概率
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # ---- 嵌入 ----
        self.token_embed = nn.Embedding(vocab_size, d_model)
        #  查表: (B, T) → (B, T, D)
        self.pos_embed = nn.Embedding(max_len, d_model)
        #  查表: (T,) → (T, D)，广播到 (B, T, D)
        self.drop = nn.Dropout(dropout)

        # ---- N 层 Block ----
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # ---- 输出 ----
        self.ln_final = nn.LayerNorm(d_model)   # 最后统一归一化
        self.head = nn.Linear(d_model, vocab_size)
        #            (B, T, D) → (B, T, V)

    # -------- 因果mask（注册为buffer，跟着模型迁移设备）--------
    def causal_mask(self, T):
        mask = torch.triu(torch.ones(T, T), diagonal=1)   # 上三角=1
        mask = mask.masked_fill(mask == 1, float('-inf'))  # 1 → -inf
        return mask

    def forward(self, x):
        """
        x: (B, T)  token id 序列
        → (B, T, V)  每个位置预测下一个字的概率
        """
        B, T = x.shape

        # ---- ① 嵌入 ----
        tok = self.token_embed(x) * math.sqrt(self.d_model)
        #   (B, T) → (B, T, D)
        #   × sqrt(D): 放大嵌入，防止嵌入太小被位置编码淹没

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        #   (1, T) → (B, T)

        x = tok + self.pos_embed(pos)
        #   (B,T,D) + (B,T,D) → (B,T,D)   逐元素相加
        x = self.drop(x)                    # (B, T, D)

        # ---- ② 过 N 层 ----
        mask = self.causal_mask(T).to(x.device)   # (T, T)
        for block in self.blocks:
            x = block(x, mask)
        #   x: (B, T, D)  — 每层输出形状不变

        # ---- ③ 输出头 ----
        x = self.ln_final(x)          # (B, T, D)
        logits = self.head(x)         # (B, T, V)
        return logits

    # -------- 生成：自回归解码 --------
    @torch.no_grad()
    def generate(self, token_ids, max_new=50, temperature=1.0):
        """
        token_ids: (1, current_len) 已生成的 token
        返回: (1, current_len + max_new)

        temperature=1.0 正常, <1.0 保守(贪心), >1.0 冒险
        """
        self.eval()
        for _ in range(max_new):
            # 截断到 max_len
            inp = token_ids[:, -self.max_len:]   # (1, ≤max_len)

            # 前向
            logits = self(inp)                    # (1, T, V)
            logits = logits[:, -1, :] / temperature  # 取最后位置 (1, V)

            # 采样下一个 token
            probs = F.softmax(logits, dim=-1)     # (1, V)
            next_id = torch.multinomial(probs, 1) # 按概率采样 (1, 1)
            #       ↑ multinomial 随机采样，不用 argmax，生成更多样

            token_ids = torch.cat([token_ids, next_id], dim=1)  # (1, T+1)
        return token_ids


# ================================================================
#  5. 维度变换一览
# ================================================================
if __name__ == "__main__":
    V, D, h, L = 100, 128, 4, 4     # 词表100, 128维, 4头, 4层
    B, T = 2, 8                     # batch=2, seq_len=8

    model = GPT(V, d_model=D, n_heads=h, n_layers=L, max_len=64)
    x = torch.randint(0, V, (B, T))

    print("=" * 55)
    print("维度变化一览（B=2, T=8, D=128, h=4, V=100, L=4）")
    print("=" * 55)
    print(f"  输入 x:          {tuple(x.shape)}")
    print(f"    → token_embed×√D: {tuple(x.shape) + (D,)}")
    print(f"    → +pos_embed:     {tuple(x.shape) + (D,)}")
    print(f"    → Dropout:        {tuple(x.shape) + (D,)}")
    print(f"    → Block×{L}:       {tuple(x.shape) + (D,)} (每层不变)")
    print(f"      ├─ ln1 → attn:  {tuple(x.shape) + (D,)}")
    print(f"      │    Q/K/V:     ({B},{h},{T},{D//h})")
    print(f"      │    scores:    ({B},{h},{T},{T})")
    print(f"      │    attn×V:    ({B},{h},{T},{D//h})")
    print(f"      │    →W_o:      ({B},{T},{D})")
    print(f"      ├─ +残差:       ({B},{T},{D})")
    print(f"      ├─ ln2 → FFN:  ({B},{T},{D*4})")
    print(f"      └─ →W2:        ({B},{T},{D})")
    print(f"    → ln_final:       {tuple(x.shape) + (D,)}")
    print(f"    → head(D→V):      ({B},{T},{V})  ← 每个位置预测词表概率")

    # 验证
    logits = model(x)
    print(f"\n  实际输出 logits: {tuple(logits.shape)}")
    print(f"  含义: 每个位置给出{V}个字的得分，取最高分的字就是预测")

    # 生成演示
    print(f"\n{'='*55}")
    print("生成演示: 输入 [0,1,2] 三个 token，生成 10 个")
    inp = torch.tensor([[0, 1, 2]])
    out = model.generate(inp, max_new=10)
    print(f"  输入: {inp.tolist()}")
    print(f"  输出: {out.tolist()}")

    # 因果mask演示
    print(f"\n{'='*55}")
    print("因果mask示例 (T=5):")
    m = model.causal_mask(5)
    for row in m:
        print("  " + " ".join("0" if v==0 else "-∞" for v in row.tolist()))
    print("  每行=该位置能看到的范围，-∞=被屏蔽的未来")
