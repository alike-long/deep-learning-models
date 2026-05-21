"""
Swin Transformer — Fashion MNIST 训练 + 测试
==============================================
模型从 swin_model.py 导入
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from swin_model import SwinTransformer


# ================================================================
#  训练 + 测试
# ================================================================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 数据 ----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.FashionMNIST("../01_data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST("../01_data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # ---- 模型 ----
    model = SwinTransformer().to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Swin Transformer — Fashion MNIST")
    print(f"参数: {sum(p.numel() for p in model.parameters()):,}\n")

    # ---- 训练循环 ----
    for epoch in range(10):
        model.train()
        total_loss = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # ---- 测试 ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)
                pred = model(img).argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

        print(f"Epoch {epoch+1:2d} | "
              f"train_loss: {total_loss/len(train_loader):.4f} | "
              f"test_acc: {correct/total:.2%}")

    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
