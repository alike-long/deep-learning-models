"""
GPT 最简实现 — 纯 Decoder 文本生成
===================================
用途：根据开头自动往下写，比如"悲伤的歌" → "悲伤的歌越唱越难过..."

和 BERT 的区别：
  GPT = Decoder Only, 单向(只看左边), 做生成
  BERT = Encoder Only, 双向(左右都看), 做理解

核心三件套：
  1. 因果掩码（上三角mask）→ 保证不能偷看未来
  2. 自回归生成 → 每次只预测下一个词，预测完拼回去再预测下下个
  3. 位置编码 → 告诉模型词的顺序
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


# ================================================================
#  第1部分：造小数据集 + 分词
# ================================================================
lyrics = """
雪下得那么深下得那么认真
我想摸你的头发只是简单的试探啊
不找了不再找了就让过去随风飘散
你还要我怎样要怎样
我想和你在一起哪怕一天也可以
遇见你是最美丽的意外
时光不老我们不散
往事不必回首余生各自安好
分开后我会笑着说没事但我也会难过
爱过你很值得我不要你怎样没怎样
想带你环游星球从日出到日落
找不到了你给过我的温柔再也找不到了
人间烟火岁岁平安
你的笑容那么温柔像春风吹过心头
"""

# 按字分词（中文最简单的分词方式）
chars = sorted(set(lyrics))
vocab = {c: i for i, c in enumerate(chars)}        # 字 → id
id2c = {i: c for c, i in vocab.items()}             # id → 字
vocab_size = len(vocab)
print(f"词表大小: {vocab_size}")

# 把歌词转成 id 序列
data = [vocab[c] for c in lyrics if c != "\n"]

# 截成多个 seq_len=20 的小段，每段 x[0:20] → y[1:21]
seq_len = 20
xs, ys = [], []
for i in range(0, len(data) - seq_len, seq_len):
    xs.append(data[i : i + seq_len])
    ys.append(data[i + 1 : i + seq_len + 1])

x_all = torch.tensor(xs, dtype=torch.long)  # (样本数, 20)
y_all = torch.tensor(ys, dtype=torch.long)
print(f"训练样本: {x_all.shape[0]} 条, 每条 {seq_len} 个字")


# ================================================================
#  第2部分：GPT 模型
# ================================================================
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3):
        super().__init__()

        # ① 词嵌入 + 位置嵌入
        # 两个都是 nn.Embedding，训练时一起学
        self.token_embed = nn.Embedding(vocab_size, d_model)   # 每个字→128维
        self.pos_embed = nn.Embedding(seq_len, d_model)        # 每个位置→128维

        # ② Decoder 层（nn.TransformerDecoderLayer 带因果mask就是GPT）
        self.dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.dec_layer, n_layers)

        # ③ 输出头：128维 → 词表大小（每个字的得分）
        self.fc = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    # -------- 因果掩码：上三角矩阵 --------
    def causal_mask(self, sz):
        """
        返回 (sz, sz) 的矩阵：
          [[0, -∞, -∞, -∞],   位置0只能看0
           [0,  0, -∞, -∞],   位置1只能看0,1
           [0,  0,  0, -∞],   位置2只能看0,1,2
           [0,  0,  0,  0]]   位置3可以看全部
        softmax(+)后，-∞变成0权重，"未来的词"被彻底屏蔽
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)  # 上三角=1, 其余=0
        mask = mask.masked_fill(mask == 1, float('-inf'))  # 1 → -inf
        return mask

    def forward(self, x):
        B, T = x.shape  # T = seq_len = 20

        # 位置编号 0~19，每个 sample 都一样
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # token + 位置 相加 → 乘以 sqrt(d_model) 防梯度消失
        x = self.token_embed(x) * math.sqrt(self.d_model)
        x = x + self.pos_embed(pos)

        # 因果 mask：每个位置只能看到自己及左边
        mask = self.causal_mask(T).to(x.device)

        # Decoder：x 当 query，memory 也用 x 自身（纯 decoder，没 encoder）
        # 因为 TransformerDecoder 需要 memory 参数（来自encoder），
        # 但 GPT 没有 encoder，所以把 x 自己填进去，并设为全0
        memory = torch.zeros(B, 1, self.d_model, device=x.device)
        x = self.decoder(x, memory, tgt_mask=mask)

        # 映射回词表大小
        return self.fc(x)  # (B, T, vocab_size)


# ================================================================
#  第3部分：训练
# ================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleGPT(vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()

    print("开始训练...")
    for epoch in range(200):
        opt.zero_grad()

        out = model(x_all.to(device))               # (样本, 20, 词表大小)
        loss = loss_fn(out.reshape(-1, vocab_size),  # 展平：CrossEntropy期望(N,类别)
                       y_all.to(device).reshape(-1)) # 展平：CrossEntropy期望(N,)

        loss.backward()
        opt.step()

        if epoch % 40 == 0:
            ppl = math.exp(loss.item())   # 困惑度：越小越好，=1是完美
            print(f"Epoch {epoch:3d} | loss:{loss.item():.3f} | 困惑度:{ppl:.1f}")

    return model, device


# ================================================================
#  第4部分：生成（自回归解码）
# ================================================================
def generate(model, device, start_text, max_new=30):
    """
    自回归生成：每次预测下一个字，拼回去再预测下下个
    就像接龙：给定"悲伤的"，预测出"歌"，再给"悲伤的歌"，预测出"越"...
    """
    model.eval()
    ids = [vocab[c] for c in start_text]        # 开头→id

    with torch.no_grad():
        for _ in range(max_new):
            # 只取最后 seq_len 个，太长会超出位置编码范围
            inp = torch.tensor([ids[-seq_len:]], device=device)
            logits = model(inp)                   # (1, T, 词表大小)
            next_id = logits[0, -1, :].argmax().item()  # 取最后位置的预测
            ids.append(next_id)

    return "".join(id2c[i] for i in ids)


# ================================================================
#  第5部分：运行
# ================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    model, device = train()

    print("\n=== 生成歌词 ===")
    prompts = ["悲伤的", "我想和", "雪下得", "不找了"]
    for p in prompts:
        print(f"\n  开头: {p}")
        print(f"  生成: {generate(model, device, p, 30)}")
