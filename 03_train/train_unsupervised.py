"""
无监督学习 — 没有标签，让模型自己发现数据中的结构
=====================================================
自编码器(AutoEncoder): 把数据压成低维，再还原回来
  输入 → Encoder → 压缩表示(瓶颈) → Decoder → 重建
  目标: 重建出来的和输入越像越好（不需要标签！）
"""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(200, 8)                      # 200条，8维（没有标签！）

# ===== 自编码器 =====
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码：8 → 4 → 2（压缩到2维）
        self.encoder = nn.Sequential(
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 2),                  # ← 瓶颈: 只用2个数表示8个数的信息
        )
        # 解码：2 → 4 → 8（从2维还原回8维）
        self.decoder = nn.Sequential(
            nn.Linear(2, 4), nn.ReLU(),
            nn.Linear(4, 8),
        )

    def forward(self, x):
        compressed = self.encoder(x)           # (B, 2) ← 压缩信息
        rebuilt = self.decoder(compressed)     # (B, 8) ← 还原
        return rebuilt, compressed

model = AutoEncoder()
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()                       # 重建损失: 和原始数据比

for epoch in range(100):
    opt.zero_grad()
    rebuilt, _ = model(X)
    loss = loss_fn(rebuilt, X)               # 重建 vs 原始（不需要y！）
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | 重建误差: {loss:.4f}")

loss_final = loss_fn(model(X)[0], X).item()
print(f"\n最终重建误差: {loss_final:.4f} （越小说明2维压缩保留了越多信息）")
print("特点: 无标签，模型自己学会数据的内在结构（压缩→还原）")
