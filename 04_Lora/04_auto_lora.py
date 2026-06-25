"""
自动挂载 — 一行 auto_lora() 给 resnet18 加 LoRA
=================================================
不看源码, named_modules() 自动递归找到所有 Linear & Conv → 挂旁路
"""
import torch, torch.nn as nn, torch.optim as optim
from torchvision.models import resnet18

torch.manual_seed(42)

class LoRALinear(nn.Module):
    def __init__(self, linear, r=2, alpha=4):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters(): p.requires_grad = False
        out_d, in_d = linear.weight.shape
        self.A = nn.Parameter(torch.randn(out_d, r) * 0.02)
        self.B = nn.Parameter(torch.randn(r, in_d) * 0.02)
        self.scale = alpha / r
    def forward(self, x):
        return self.linear(x) + (x @ self.B.T @ self.A.T) * self.scale

class ConvLoRA(nn.Module):
    def __init__(self, conv, r=2, alpha=4):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters(): p.requires_grad = False
        self.scale = alpha / r
        self.down = nn.Conv2d(conv.in_channels, r, 1, bias=False)
        self.up   = nn.Conv2d(r, conv.out_channels, 1, bias=False)
        nn.init.normal_(self.down.weight, std=0.02); nn.init.zeros_(self.up.weight)
    def forward(self, x):
        return self.conv(x) + self.up(self.down(x)) * self.scale

def auto_lora(model, r=2, alpha=4):
    count = 0
    for name, module in model.named_modules():
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        child_name  = name.rsplit('.', 1)[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        if isinstance(module, nn.Linear):
            setattr(parent, child_name, LoRALinear(module, r, alpha)); count += 1
        elif isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            setattr(parent, child_name, ConvLoRA(module, r, alpha)); count += 1
    # 只训 LoRA 参数
    for n, p in model.named_parameters():
        is_lora = any(k in n for k in ['A', 'B', 'down', 'up'])
        p.requires_grad = is_lora
    return model, count

# ---- 演示 ----
model = resnet18(weights=None); model.fc = nn.Linear(512, 10)
total = sum(p.numel() for p in model.parameters())
model, n = auto_lora(model, r=4, alpha=8)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数: {total:,} | 挂{n}层 | 可训: {trainable:,} ({trainable/total:.1%})")

X, Y = torch.randn(16, 3, 224, 224), torch.randint(0, 10, (16,))
opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
for _ in range(3):
    opt.zero_grad(); loss = nn.CrossEntropyLoss()(model(X), Y); loss.backward(); opt.step()
print(f"训练完成, loss={loss.item():.4f}")
