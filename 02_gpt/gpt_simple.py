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
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ↑ 确保从任意目录运行都能找到同目录下的 gpt_full_model

import torch
import math
import torch.nn as nn
from gpt_full_model import GPT


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
#  第2部分：GPT 模型（直接用 gpt_full_model 的完整 GPT）
# ================================================================
# 不用 SimpleGPT 了，直接用手写的 GPT（CausalSelfAttention + Block × N）
# 导入语句已在文件顶部: from gpt_full_model import GPT


# ================================================================
#  第3部分：训练
# ================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(vocab_size, max_len=seq_len).to(device)   # ★ 模型移到 GPU
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
    # 如果字不在词表，跳过（避免 KeyError）
    ids = [vocab[c] for c in start_text if c in vocab]
    if not ids:
        return "(开头词语都不在词表中)"

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
