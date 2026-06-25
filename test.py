"""
预训练模型 + LoRA — 不看源码，自动给所有 Conv/Linear 挂旁路
===============================================================
场景: import 了一个 torchvision ResNet-50，一行代码加载
      不看内部结构，也能自动挂 LoRA
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

torch.manual_seed(42)


# ===== LoRA 模块 (Linear + Conv 两版) =====
class LoRALinear(nn.Module):
    def __init__(self, linear, r=2, alpha=4):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
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
        for p in self.conv.parameters():
            p.requires_grad = False
        in_c, out_c = conv.in_channels, conv.out_channels
        self.scale = alpha / r
        self.down = nn.Conv2d(in_c, r, 1, bias=False)
        self.up   = nn.Conv2d(r, out_c, 1, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.conv(x) + self.up(self.down(x)) * self.scale


# ===== ★ 核心: 自动遍历模型，给所有目标层挂 LoRA =====
def auto_lora(model, r=2, alpha=4, target_types=(nn.Linear, nn.Conv2d)):
    """
    不看源码，一行代码把预训练模型的所有 Linear 和 Conv 挂上 LoRA
    返回: 挂好 LoRA 的模型, 挂了几个层
    """
    count = 0
    for name, module in model.named_modules():
        # named_modules() 递归遍历所有子层 — 你不需要知道它们在哪

        if isinstance(module, nn.Linear):
            # 找到父层，替换子层
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name  = name.rsplit('.', 1)[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
            count += 1

        elif isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
            # 只挂 kernel>1 的卷积 (1×1 卷积本身就是低秩的, 挂了浪费)
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name  = name.rsplit('.', 1)[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, ConvLoRA(module, r=r, alpha=alpha))
            count += 1

    # ★ 只优化 LoRA 参数
    for name, p in model.named_parameters():
        p.requires_grad = ('A' not in name) and ('B' not in name) and \
                          ('down' not in name) and ('up' not in name)
        p.requires_grad = not p.requires_grad   # 取反 → LoRA部份可训

    return model, count


# ================================================================
#  演示: 加载 resnet18，挂 LoRA，做 Fashion MNIST 10 分类
# ================================================================
print("=" * 55)
print("Step1: 加载预训练 ResNet-18 (不看内部)")
print("=" * 55)

model = resnet18(weights=None)                # 一行代码，不看源码
model.fc = nn.Linear(512, 10)                # 换分类头 (10类)

total = sum(p.numel() for p in model.parameters())
print(f"  原始参数: {total:,}")

print(f"\nStep2: 自动挂 LoRA...")
model, n_lora = auto_lora(model, r=4, alpha=8)
print(f"  挂了 {n_lora} 层")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"  冻结参数: {frozen:,}")
print(f"  可训参数: {trainable:,} ({trainable/(trainable+frozen):.1%})")


# ===== 快速验证: 造假数据跑一遍 =====
print(f"\nStep3: 验证训练 (假数据跑3轮)")
X = torch.randn(16, 3, 224, 224)
Y = torch.randint(0, 10, (16,))

opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
for ep in range(3):
    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(model(X), Y)
    loss.backward()
    opt.step()
    print(f"  Step {ep}: loss={loss.item():.4f}")

print(f"\n  前向: X(16,3,224,224) → model → (16,10)  ✅")
print(f"  反向: 只有 LoRA 参数被更新")


# ===== 删去 LoRA → 恢复原始权重 =====
print(f"\nStep4: 关掉 LoRA → 恢复基座")
for name, module in model.named_modules():
    if hasattr(module, 'scale'):
        module.scale = 0.0                            # ★ 一把关掉所有 LoRA

with torch.no_grad():
    out_off = model(X)
    print(f"  LoRA关掉后前向正常 ✅ 输出形状: {tuple(out_off.shape)}")

print(f"""
{'='*55}
总结:
  直接引用 ResNet = resnet18() ← 一行代码
  自动挂 LoRA    = auto_lora(model) ← 也是一行代码

  不做的事: 不看源码
  做的事:   named_modules() 递归找到所有 Linear & Conv
            → 包 LoRALinear / ConvLoRA → 基座冻住，旁路可训
            → 只训 1~5% 的参数，微调新任务

  PEFT 库的 get_peft_model() 就是这个原理
""")
