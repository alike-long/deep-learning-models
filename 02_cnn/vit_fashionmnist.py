import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 超参
batch_size = 128
d_model = 64
num_heads = 2
num_layers = 2
img_size = 28
patch_size = 4
num_classes = 10

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练/测试集
train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Encoder层（无掩码，双向注意力）
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model)
        )
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.fc(x))
        return x

# Encoder 分类模型
class TransformerEncoderCls(nn.Module):
    def __init__(self, d_model, nhead, num_layers, patch_num, num_classes):
        super().__init__()
        self.proj = nn.Linear(patch_size*patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, patch_num)
        self.encoder_blocks = nn.Sequential(*[
            EncoderLayer(d_model, nhead) for _ in range(num_layers)
        ])
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        # 图片分patch变成序列
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        x = x.contiguous().view(B, -1, patch_size*patch_size)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.encoder_blocks(x)
        x = x.mean(dim=1)
        return self.cls_head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_num = (img_size // patch_size) ** 2
model = TransformerEncoderCls(d_model, num_heads, num_layers, patch_num, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 测试函数
def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            out = model(img)
            _, pred = torch.max(out, dim=1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    return correct / total

# 训练+每轮测试
for epoch in range(5):
    model.train()
    total_loss = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    test_acc = test(model, test_loader)
    print(f"Epoch [{epoch+1}/5] | Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")