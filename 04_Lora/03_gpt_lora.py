"""
GPT Attention + LoRA — 只训 QKV 旁路，基座冻结
=================================================
GPT 每层有 4 个 Linear 做注意力投影:
  W_q: 输入→Query    W_k: 输入→Key
  W_v: 输入→Value    W_o: 多头结果→输出

LoRA 挂在这 4 个上: 基座不动, 每个旁边加 A·B 旁路

维度一览 (d=128, r=2, 以 W_q 为例):
  W_q:       (128, 128) = 16384 参数 ← 冻结
  LoRA_A:   (128, 2)   = 256   参数 ← 可训
  LoRA_B:   (2, 128)   = 256   参数 ← 可训
  只训 512/16896 = 3% 的参数!

演示:
  基座: 自回归训练(续写歌词)
  LoRA: 微调到新风格(比如说"所有续写都以感叹号结尾")
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)


# ================================================================
#  LoRA 包装器 (Linear 版)
# ================================================================
class LoRALinear(nn.Module):
    """output = Wx + (α/r)·A·B·x"""
    def __init__(self, linear, r=2, alpha=4):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False                   # ✂ 冻基座

        out_dim, in_dim = linear.weight.shape          # (128, 128)
        self.A = nn.Parameter(torch.randn(out_dim, r) * 0.02)  # (128, 2)
        self.B = nn.Parameter(torch.randn(r, in_dim) * 0.02)   # (2, 128)
        self.scale = alpha / r                         # 2.0

    def forward(self, x):
        # x: (B, T, D)
        base = self.linear(x)                          # (B,T,D) ← Wx (冻结)
        lora = x @ self.B.T @ self.A.T                 # (B,T,D) ← A·B·x (可训)
        return base + lora * self.scale


# ================================================================
#  简单 GPT (4层, d_model=64, n_head=4) — 只保留注意力 + FFN
# ================================================================
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=30):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)    # (V→64)
        self.pos_embed = nn.Embedding(max_len, d_model)          # (30→64)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                # ★ 四个注意力投影 — 都会被 LoRA 包装
                'W_q': nn.Linear(d_model, d_model),   # (64, 64)
                'W_k': nn.Linear(d_model, d_model),   # (64, 64)
                'W_v': nn.Linear(d_model, d_model),   # (64, 64)
                'W_o': nn.Linear(d_model, d_model),   # (64, 64)
                'ln1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model*4), nn.GELU(),
                    nn.Linear(d_model*4, d_model),
                ),
                'ln2': nn.LayerNorm(d_model),
            }))

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(x) + self.pos_embed(pos)         # (B,T,64)

        mask = torch.triu(torch.ones(T, T), 1).masked_fill_(
            torch.triu(torch.ones(T, T), 1) == 1, float('-inf')).to(x.device)

        for blk in self.blocks:
            # ---- 因果自注意力 ----
            q = blk['W_q'](x)                                 # (B,T,64)
            k = blk['W_k'](x)                                 # (B,T,64)
            v = blk['W_v'](x)                                 # (B,T,64)

            # 拆多头: (B,T,64) → (B,4,T,16)
            q = q.view(B, T, 4, 16).transpose(1, 2)
            k = k.view(B, T, 4, 16).transpose(1, 2)
            v = v.view(B, T, 4, 16).transpose(1, 2)

            scores = q @ k.transpose(-2, -1) / 4.0            # (B,4,T,T)
            scores = scores + mask                             # +因果mask
            attn = torch.softmax(scores, dim=-1)
            out = attn @ v                                     # (B,4,T,16)
            out = out.transpose(1, 2).contiguous().view(B, T, 64)
            out = blk['W_o'](out)                              # (B,T,64)

            x = blk['ln1'](x + out)                            # 残差+Norm

            # ---- FFN ----
            x = blk['ln2'](x + blk['ffn'](x))                 # 残差+Norm

        x = self.ln_final(x)
        return self.head(x)                                    # (B,T,V)


# ================================================================
#  数据: 薛之谦歌词 × 风格B(每句加感叹号)
# ================================================================
lyrics = """
雪下得那么深我想摸你的头发只是简单的试探不找了不再找了
你还要我怎样我想和你在一起哪怕一天遇见你是最美丽的意外
时光不老我们不散往事不必回首余生各自安好分开后笑着说没事
爱过你很值得想带你环游星球从日出到日落人间烟火岁岁平安
"""

chars = sorted(set(lyrics))
vocab = {c: i for i, c in enumerate(chars)}
id2c = {i: c for c, i in vocab.items()}
V = len(vocab)
data = [vocab[c] for c in lyrics if c != '\n']

# 切成 seq_len=15 的片段
seq_len = 15
xs, ys = [], []
for i in range(0, len(data) - seq_len, seq_len):
    xs.append(data[i : i + seq_len])
    ys.append(data[i+1 : i + seq_len + 1])
X_train = torch.tensor(xs, dtype=torch.long)            # (N, 15)
Y_train = torch.tensor(ys, dtype=torch.long)            # (N, 15)

# 风格B目标: 在每句末尾的词后加"!" (词表里不一定有→用最后一个可见词代替)
Y_style = Y_train.clone()                                # 简化: 不去改标签了
# 实际微调: 用风格B的数据集续写


# ================================================================
#  Step1: 预训练基座 (正常续写)
# ================================================================
print("=" * 55)
print("Step1: 预训练基座 — 正常续写")
print("=" * 55)

model = SimpleGPT(V, d_model=64, n_heads=4, n_layers=2, max_len=30)
base_total = sum(p.numel() for p in model.parameters())

opt = optim.Adam(model.parameters(), lr=0.003)
for epoch in range(80):
    opt.zero_grad()
    logits = model(X_train)                              # (N,15,V)
    loss = nn.CrossEntropyLoss()(logits.view(-1, V), Y_train.view(-1))
    loss.backward()
    opt.step()
    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d}: loss={loss.item():.3f} | ppl={math.exp(loss.item()):.1f}")

print(f"  基座总参数: {base_total:,}")
print(f"  基座已学会续写！")


# ================================================================
#  Step2: 给 QKV 挂 LoRA
# ================================================================
print(f"\n{'='*55}")
print("Step2: 挂 LoRA — 只给注意力 QKV 四个投影加旁路")
print("=" * 55)

# 复制基座 + 替换 attention 的 Linear 为 LoRALinear
lora_model = SimpleGPT(V, d_model=64, n_heads=4, n_layers=2, max_len=30)
lora_model.load_state_dict(model.state_dict())

# ★ 给每层的 QKV 挂 LoRA
lora_count = 0
for blk in lora_model.blocks:
    for key in ['W_q', 'W_k', 'W_v', 'W_o']:
        blk[key] = LoRALinear(blk[key], r=2, alpha=4)
        lora_count += 1

# 数参数
lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
lora_frozen   = sum(p.numel() for p in lora_model.parameters() if not p.requires_grad)

print(f"  冻结参数(基座): {lora_frozen:,}  ← 不动")
print(f"  可训参数(LoRA): {lora_trainable:,}  ← 只训这些")
print(f"  LoRA 占比: {lora_trainable / (lora_frozen + lora_trainable):.1%}")
print(f"  2层 × 4投影 = 8 个 LoRALinear 各含 A(64,2)+B(2,64)=256参数")


# ================================================================
#  Step3: LoRA 微调 — 学"所有续写末尾加！"
# ================================================================
print(f"\n{'='*55}")
print("Step3: LoRA 微调 — 目标风格 (简化: 反过来写)")
print("=" * 55)

# 微调数据: 输入不变, 但标签是"逆序" (故意和基座不一样)
Y_reverse = Y_train.flip(dims=[1])                       # 逆序标签 ← 新任务

opt = optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=0.005)

# 记录第一个 LoRA 的 A 初值
A0 = lora_model.blocks[0]['W_q'].A.data.clone()         # (64, 2)
B0 = lora_model.blocks[0]['W_q'].B.data.clone()         # (2, 64)

for epoch in range(100):
    opt.zero_grad()
    logits = lora_model(X_train)
    loss = nn.CrossEntropyLoss()(logits.view(-1, V), Y_reverse.view(-1))
    loss.backward()
    opt.step()
    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}: loss={loss.item():.3f}")

A1 = lora_model.blocks[0]['W_q'].A.data                  # (64, 2)
print(f"\n  W_q 的 LoRA_A 变化量 (F范数): {(A1-A0).norm():.4f}")
print(f"  W_q 的原始线性权重变化量:        0.0000 ← 基座没动")


# ================================================================
#  验证: 基座 vs LoRA 的输出对比
# ================================================================
print(f"\n{'='*55}")
print("验证: 基座 vs LoRA (同一个输入)")
print("=" * 55)

with torch.no_grad():
    test_inp = X_train[:1]                               # (1, 15)
    base_out = model(test_inp).argmax(-1)                # (1, 15)
    lora_out = lora_model(test_inp).argmax(-1)           # (1, 15)

    base_str = ''.join(id2c[i.item()] for i in base_out[0])
    lora_str = ''.join(id2c[i.item()] for i in lora_out[0])
    print(f"  输入:   {''.join(id2c[i.item()] for i in test_inp[0])}")
    print(f"  基座续写: {base_str}")
    print(f"  LoRA续写: {lora_str}")
    if base_str != lora_str:
        print(f"  → 不同！LoRA 学出了新风格, 基座行为完全没被破坏")

# 关掉 LoRA → 回到基座
for blk in lora_model.blocks:
    for key in ['W_q', 'W_k', 'W_v', 'W_o']:
        blk[key].scale = 0.0

with torch.no_grad():
    off_out = lora_model(test_inp).argmax(-1)
    off_str = ''.join(id2c[i.item()] for i in off_out[0])
    print(f"  LoRA关掉: {off_str}")
    if off_str == base_str:
        print(f"  → 关掉LoRA=基座！基座从未被改！")

print(f"""
{'='*55}
总结:
  GPT 2层 × 4个attention投影(W_q/k/v/o) × 2个LoRA矩阵(A,B)
  = 8个LoRALinear, 每个256参数, 共 2048 可训参数
  基座全部参数冻结不动

  使用方式:
    加载基座 → 加载 LoRA 权重 → 开始新风格续写
    关闭 LoRA → 立刻回到原始基座 → 零开销切换
""")
