"""
半监督学习 — 少量有标签 + 大量无标签 混合训练
=================================================
方法: 伪标签 (Pseudo-Labeling)
  ① 先用有标签的数据训一个模型
  ② 用模型预测无标签数据，把高置信度的预测当"真标签"
  ③ 把伪标签数据加回训练集，继续训
"""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# ===== 数据: 只有20条有标签，80条无标签 =====
X_labeled = torch.randn(20, 4)               # 20条有标签
y_labeled = (X_labeled.sum(dim=1) > 0).long()

X_unlabeled = torch.randn(80, 4)             # 80条没标签！

model = nn.Linear(4, 2)                      # 最简单的二分类
opt = optim.Adam(model.parameters(), lr=0.02)
loss_fn = nn.CrossEntropyLoss()

# ===== 阶段1: 只拿有标签的数据训 =====
print("阶段1: 只用有标签数据（20条）")
for epoch in range(50):
    opt.zero_grad()
    loss = loss_fn(model(X_labeled), y_labeled)
    loss.backward()
    opt.step()
    if epoch % 49 == 0:
        acc = (model(X_labeled).argmax(dim=1) == y_labeled).sum().item() / 20
        print(f"  有标签准确率: {acc:.0%}")

# ===== 阶段2: 用模型给无标签数据打"伪标签" =====
with torch.no_grad():
    logits = model(X_unlabeled)               # 给80条无标签数据预测
    probs = torch.softmax(logits, dim=1)       # 转成概率
    confidence, pseudo_y = probs.max(dim=1)    # 取最高分的类别

    # 只保留置信度 > 0.8 的（认为预测比较靠谱）
    high_conf = confidence > 0.8
    X_pseudo = X_unlabeled[high_conf]
    y_pseudo = pseudo_y[high_conf]
    print(f"\n阶段2: 高置信度伪标签: {len(X_pseudo)} 条 (共 80 条无标签中)")

# 把伪标签数据跟原标签数据拼一起
X_all = torch.cat([X_labeled, X_pseudo])
y_all = torch.cat([y_labeled, y_pseudo])

print(f"阶段3: 合并训练 ({len(X_all)} 条)")
for epoch in range(50):
    opt.zero_grad()
    loss = loss_fn(model(X_all), y_all)
    loss.backward()
    opt.step()

acc = (model(X_all).argmax(dim=1) == y_all).sum().item() / len(y_all)
print(f"最终准确率: {acc:.0%}")
print("特点: 省标注成本，用模型自己打伪标签扩充训练集")
