"""
MHA + LoRA — 手写 MultiheadAttention，QKV 各自挂 LoRA
=======================================================
官方 nn.MultiheadAttention 的 QKV 在 in_proj_weight(3D,D) 拆不开
→ 手写一份 MHA，QKV 各是独立 Linear → 直接包 LoRALinear
"""
import torch, torch.nn as nn

torch.manual_seed(42)
D, H = 64, 4

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

class MultiheadAttentionLoRA(nn.Module):
    def __init__(self, d_model=64, n_heads=4, r=2, alpha=4):
        super().__init__()
        self.n_heads = n_heads; self.d_k = d_model // n_heads
        # ★ QKV 各独立 — 想挂 LoRA 就挂，不想挂的留 nn.Linear
        self.W_q = LoRALinear(nn.Linear(d_model, d_model), r, alpha)
        self.W_k = LoRALinear(nn.Linear(d_model, d_model), r, alpha)
        self.W_v = LoRALinear(nn.Linear(d_model, d_model), r, alpha)
        self.W_o = nn.Linear(d_model, d_model)   # O 不挂

    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        scores = Q @ K.transpose(-2,-1) / (self.d_k**0.5)
        if mask is not None: scores = scores + mask
        out = torch.softmax(scores, dim=-1) @ V
        out = out.transpose(1,2).contiguous().view(B, T, D)
        return self.W_o(out)

# 验证
mha = MultiheadAttentionLoRA(D, H)
x = torch.randn(2, 8, D)
out = mha(x)
trainable = sum(p.numel() for p in mha.parameters() if p.requires_grad)
frozen   = sum(p.numel() for p in mha.parameters() if not p.requires_grad)
print(f"输出形状: {tuple(out.shape)}  |  可训: {trainable}  |  冻结: {frozen}")
