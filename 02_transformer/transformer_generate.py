import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------- 读取数据集（和你原来完全一样） ---------------------
with open(r"F:\py-code\vection-work\jay_lyrics.txt.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# --------------------- 构建字典（完全不变） ---------------------
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}
data_idx = [char2idx[c] for c in text]

# --------------------- 构建训练序列（完全不变） ---------------------
seq_len = 16
xs, ys = [], []
for i in range(len(data_idx) - seq_len):
    xs.append(data_idx[i:i + seq_len])
    ys.append(data_idx[i + 1:i + seq_len + 1])

x = torch.tensor(xs, dtype=torch.long)
y = torch.tensor(ys, dtype=torch.long)


# --------------------- 位置编码（你刚才学的！） ---------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


# --------------------- 掩码 Softmax（你刚学的！） ---------------------
def masked_softmax(X, valid_lens=None):
    if valid_lens is None:
        return F.softmax(X, dim=-1)
    mask = torch.arange(X.shape[-1], device=X.device)[None, :] < valid_lens[:, None]
    while mask.dim() < X.dim():
        mask.unsqueeze_(1)
    X = X.masked_fill(~mask, -1e6)
    return F.softmax(X, dim=-1)


# --------------------- Transformer 解码器层（专门做文本生成） ---------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # 自注意力 + 掩码
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 前馈网络
        ff_out = self.fc2(F.relu(self.fc1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        return x


# --------------------- Transformer 模型（替换你的 LSTM） ---------------------
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=2, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)  # 跟 LSTM 一样
        self.pos_encoder = PositionalEncoding(d_model)  # 必须加位置编码

        # 堆叠多层解码器
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_head) for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)  # 输出层（一样）

    # 生成上三角掩码：防止看到未来的字（核心！）
    def generate_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        # 1. 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # 2. 掩码（不让模型提前看到后面的字）
        seq_len = x.shape[1]
        mask = self.generate_mask(seq_len).to(x.device)

        # 3. 过 Transformer 层
        for layer in self.layers:
            x = layer(x, mask=mask)

        # 4. 预测下一个词
        return self.fc(x)


# --------------------- 初始化模型（和你原来一样） ---------------------
model = TransformerModel(vocab_size, d_model=64, n_head=2, num_layers=2)

# --------------------- 训练（完全不变！） ---------------------
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("开始训练...")
for epoch in range(200):
    opt.zero_grad()
    out = model(x)
    loss = criterion(out.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    opt.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | 损失: {loss.item():.3f} | 困惑度: {math.exp(loss.item()):.3f}")


# --------------------- 生成歌词（完全不变！） ---------------------
def generate(start, gen_len=100):
    model.eval()
    res = start
    with torch.no_grad():
        for _ in range(gen_len):
            inputs = [char2idx[c] for c in res[-10:]]
            inputs = torch.tensor([inputs])
            out = model(inputs)
            idx = torch.argmax(out[0, -1]).item()
            res += idx2char[idx]
    return res


# --------------------- 测试（完全不变！） ---------------------
print("\n=== Transformer 生成歌词 ===")
while True:
    a = input("输入开头：")
    if a == "-1":
        break
    print("生成：", generate(a, 20))