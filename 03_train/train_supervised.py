"""
监督学习 — 每个 x 都知道标准答案 y
=====================================
x → 模型 → 预测 → 和y比 → loss → 更新
"""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(100, 4)                      # 100条，4个特征
y = (X.sum(dim=1) > 0).long()                # 标签: 和>0为1，否则0

model = nn.Linear(4, 2)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    if epoch % 25 == 0:
        print(f"Epoch {epoch:2d} | loss: {loss:.4f}")

acc = (model(X).argmax(dim=1) == y).sum().item() / len(y)
print(f"准确率: {acc:.0%}")
print("特点: 有标签，直接学 x→y 的映射")
