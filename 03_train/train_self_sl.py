"""
自监督预训练 — 标签来自数据本身，零人工标注
==============================================
方法: MLM — 遮住词让模型猜

  原文: [我, 爱, 你, 像, 风]
  遮住: [我, MASK, 你, MASK, 风]   ← 自动生成，免费
  标签: [*,  爱,  *,  像,  *]     ← 原文就是标签
"""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# ===== 数据: 10句话，每句4个字 =====
vocab_size = 10                              # 10个字(id 0~9)
MASK_ID = 10                                 # [MASK]的id
sentences = torch.randint(0, vocab_size, (10, 4))  # (10, 4)

# ===== 自监督造标签 =====
x_masked = sentences.clone()                 # 输入：遮住部分字
y_labels = torch.full_like(sentences, -100)  # 标签：只有被遮处有值(-100=跳过)

for i in range(10):
    for j in range(4):
        if torch.rand(1).item() < 0.3:       # 30% 的字遮住
            y_labels[i, j] = sentences[i, j] # 存真实标签
            x_masked[i, j] = MASK_ID         # 换成[MASK]

# ===== 模型: Embedding → Linear =====
model = nn.Sequential(
    nn.Embedding(11, 8),                     # 11字(含[MASK])→8维
    nn.Flatten(),                            # (B,4,8)→(B,32)
    nn.Linear(32, vocab_size * 4),           # (B,40) → 每位置10类
)

opt = optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # 自动跳过-100

for epoch in range(150):
    opt.zero_grad()
    out = model(x_masked).view(-1, 4, vocab_size)  # (10,4,10)
    loss = loss_fn(out.view(-1, vocab_size), y_labels.view(-1))
    loss.backward()
    opt.step()
    if epoch % 75 == 0:
        print(f"Epoch {epoch:3d} | loss: {loss:.4f}")

# ===== 验证 =====
with torch.no_grad():
    pred = model(x_masked).view(-1, 4, vocab_size).argmax(dim=-1)
    mask = (y_labels != -100)
    correct = (pred[mask] == y_labels[mask]).sum().item()

print(f"\n原句: {sentences[0].tolist()}")
print(f"遮住: {x_masked[0].tolist()}")
print(f"猜回: {pred[0].tolist()}")
print(f"[MASK]猜对: {correct}/{mask.sum().item()}")
print("特点: 标签=原文本身，零人工成本，大规模预训练的核心方式")
