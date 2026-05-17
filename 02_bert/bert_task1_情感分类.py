"""
BERT 微调 — 任务1：情感分类（句子级）
========================================
这是 BERT 最经典的微调方式：
  - 输入一整句 → 取 [CLS] 位置的输出 → 接一个 Linear 分类 → 输出标签
  - [CLS] 被训练成整个句子的"摘要"，句子级任务只用它
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===================== 0. 导入我们之前写的 BERT =====================
from bert_model import BERTModel


# ===================== 1. 造数据集 =====================
# 标签：0=伤感 1=深情 2=释怀 3=温暖
LABELS = ["伤感", "深情", "释怀", "温暖"]

raw_data = [
    ("雪下得那么深下得那么认真", 0),
    ("我想摸你的头发只是简单的试探啊", 1),
    ("不找了不再找了就让过去随风飘散", 2),
    ("我想和你在一起哪怕一天也可以", 3),
    ("你还要我怎样要怎样你突然来的短信就够我悲伤", 0),
    ("爱过你很值得我不要你怎样没怎样", 1),
    ("往事不必回首余生各自安好", 2),
    ("遇见你是最美丽的意外", 3),
    ("找不到了你给过我的温柔再也找不到了", 0),
    ("你像天外来物一样失而复得", 1),
    ("后来我的生活还算理想没为你落到孤单的下场", 2),
    ("时光不老我们不散", 3),
    ("分开后我会笑着说没事但我也会难过", 0),
    ("想带你环游星球从日出到日落", 1),
    ("就此别过不必再说谁对谁错", 2),
    ("你的笑容那么温柔像春风吹过心头", 3),
    ("我像个哑巴说不出心里话把所有委屈都吞下", 0),
    ("我忍不住从背后抱了一下尺度掌握在不能说想你啊", 1),
    ("所有遗憾都是成全各自天涯各自安好", 2),
    ("人间烟火岁岁平安愿有人陪你走过漫长岁月", 3),
]


# ===================== 2. 简易分词器 =====================
# 真正工程里用 transformers 库的 BertTokenizer，这里自己写一个方便理解
class SimpleTokenizer:
    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.vocab = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"] + chars
        self.c2i = {c: i for i, c in enumerate(self.vocab)}     # 字→id
        self.i2c = {i: c for c, i in self.c2i.items()}          # id→字
        self.vocab_size = len(self.vocab)
        self.pad_id = self.c2i["[PAD]"]
        self.cls_id = self.c2i["[CLS]"]
        self.sep_id = self.c2i["[SEP]"]

    def encode(self, text, max_len=50):
        """把一句话变成 token id 序列：[CLS] + 每个字 + [SEP] + [PAD]补满"""
        ids = [self.cls_id]
        for c in text[:max_len - 2]:
            ids.append(self.c2i.get(c, self.c2i["[UNK]"]))
        ids.append(self.sep_id)
        ids += [self.pad_id] * (max_len - len(ids))   # 不够 max_len 就补 [PAD]
        return ids[:max_len]


# ===================== 3. Dataset =====================
class LyricsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=50):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        # encode: 字 → id → [CLS] [id1] [id2] ... [SEP] [0] [0] ...
        input_ids = torch.tensor(self.tok.encode(text, self.max_len), dtype=torch.long)
        #                       input_ids: (max_len=50,)  ← 每个样本固定50长度
        mask = (input_ids == self.tok.pad_id)
        #  mask: (50,)  ← True=pad位置，False=真实token
        return input_ids, mask, label
        #  DataLoader 打包后:
        #    input_ids: (batch=8, 50)
        #    mask:      (batch=8, 50)
        #    label:     (8,)  ← 每个样本1个标签


# ===================== 4. 分类模型 =====================
# 关键：在 BERT encoder 上面加一个 Linear，只取 [CLS] 的输出
class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=4):
        super().__init__()
        # 直接复用我们写好的 BERTModel
        self.bert = BERTModel(vocab_size, d_model=128, n_heads=4,
                              n_layers=3, d_ff=512, max_len=50, dropout=0.1)
        # ★ 句子级任务：只需要 [CLS] 的 pooled 向量（d_model=128维），接一个 Linear 出类别
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_ids, key_padding_mask=None):
        # input_ids: (batch=8, 50),  key_padding_mask: (8, 50)
        #
        # == BERT 内部维度变化 ==
        # 1. BERTEmbedding:
        #    token_embed: (8, 50) → (8, 50, 128)   每个字→128维向量
        #    pos_embed:   (8, 50) → (8, 50, 128)   位置0~49各128维
        #    seg_embed:   (8, 50) → (8, 50, 128)   全0（无分段）
        #    → 三者相加 → LayerNorm → Dropout → x: (8, 50, 128)
        #
        # 2. 过3层 BERTLayer（每层先用 attention 看上下文字，再 FFN 变换）:
        #    x: (8, 50, 128) → attention → (8, 50, 128)
        #                    → +残差 +Norm → (8, 50, 128)
        #                    → FFN(128→512→128) → +残差 +Norm → (8, 50, 128)
        #    经过3层后  x: (8, 50, 128)  形状不变
        #
        # 3. BERTPooler:
        #    x[:, 0, :] → (8, 128)   取 [CLS] 位置
        #    Linear(128→128) → Tanh → pooled: (8, 128)
        #
        _, pooled = self.bert(input_ids, key_padding_mask=key_padding_mask)
        #                               pooled: (8, 128)  ← 整句的摘要向量
        #
        # == 分类头 ==
        #    Linear(128 → 4) → logits: (8, 4)  ← 4类: 伤感/深情/释怀/温暖
        logits = self.classifier(pooled)
        return logits


# ===================== 5. 训练 =====================
def train():
    texts = [t for t, _ in raw_data]
    tokenizer = SimpleTokenizer(texts)
    dataset = LyricsDataset(raw_data, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = BERTClassifier(tokenizer.vocab_size)

    # ============ 分层学习率（核心！）============
    # 不同层给不同 lr：底层小，顶层大，分类头最大
    #
    # BERT 参数结构（命名来自 bert_model.py）:
    #   bert.embedding.*            → 底层：token/pos/seg embedding
    #   bert.layers.0.*             → 第0层 Encoder（最底层）
    #   bert.layers.1.*             → 第1层
    #   bert.layers.2.*             → 第2层（最顶层）
    #   bert.pooler.*               → Pooler
    #   classifier.*                → 分类头（全新）
    #
    # 策略: 分类头 lr=1e-3, 顶层 lr=5e-4, 中层 lr=2e-4, 底层 lr=1e-4
    #       衰减因子 0.5: 每往下一层 lr 乘以 0.5

    lr_top = 1e-3                           # 分类头：最快
    lr_decay = 0.5                          # 每往下一层乘 0.5

    param_groups = []                       # PyTorch 的分组参数列表

    # 1. 分类头 + Pooler：lr 最大
    param_groups.append({
        'params': list(model.classifier.parameters()) + list(model.bert.pooler.parameters()),
        'lr': lr_top,
        'name': 'classifier+pooler'
    })

    # 2. BERT 各层：从顶层→底层，lr 逐层衰减
    num_layers = len(model.bert.layers)     # 3
    for layer_idx in range(num_layers - 1, -1, -1):  # 2, 1, 0（从顶到底）
        decay_times = num_layers - layer_idx          # 顶层乘0次, 底层乘最多
        cur_lr = lr_top * (lr_decay ** decay_times)
        param_groups.append({
            'params': model.bert.layers[layer_idx].parameters(),
            'lr': cur_lr,
            'name': f'encoder_layer_{layer_idx}'
        })

    # 3. Embedding：lr 最小，接近冻结
    param_groups.append({
        'params': model.bert.embedding.parameters(),
        'lr': lr_top * (lr_decay ** (num_layers + 1)),  # 底到底，衰减最狠
        'name': 'embedding'
    })

    opt = torch.optim.Adam(param_groups)
    print("分层学习率设置：")
    for g in param_groups:
        print(f"  {g['name']:20s} → lr={g['lr']:.6f}")

    loss_fn = nn.CrossEntropyLoss()

    print(f"词表: {tokenizer.vocab_size} | 样本: {len(dataset)}")
    for epoch in range(50):
        total_loss = 0
        for input_ids, mask, label in loader:
            # input_ids: (8, 50)  mask: (8, 50)  label: (8,)
            opt.zero_grad()
            logits = model(input_ids, key_padding_mask=mask)
            #                     logits: (8, 4) ← 每个样本4类得分
            loss = loss_fn(logits, label)
            #      CrossEntropyLoss: (8,4) vs (8,) → 标量
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            acc = eval_acc(model, tokenizer)
            print(f"Epoch {epoch:2d} | loss:{total_loss:.2f} | 准确率:{acc:.0%}")

    return model, tokenizer


# ===================== 6. 评估 =====================
def eval_acc(model, tokenizer):
    model.eval()
    correct = 0
    for text, label in raw_data:
        ids = torch.tensor([tokenizer.encode(text)]).long()
        #    ids: (1, 50)  ← 加 batch 维
        mask = (ids == tokenizer.pad_id)
        #    mask: (1, 50)
        with torch.no_grad():
            logits = model(ids, key_padding_mask=mask)
            #           logits: (1, 4)
            pred = logits.argmax(dim=1).item()
            #              argmax → (1,) → 取标量，比如 0
        correct += (pred == label)
    model.train()
    return correct / len(raw_data)


# ===================== 7. 预测单句 =====================
def predict(model, tokenizer, text):
    model.eval()
    ids = torch.tensor([tokenizer.encode(text)]).long()
    #    ids: (1, 50)  ← 单句加 batch 维
    mask = (ids == tokenizer.pad_id)
    #    mask: (1, 50)
    with torch.no_grad():
        logits = model(ids, key_padding_mask=mask)
        #           logits: (1, 4) ← 4类得分
        pred = logits.argmax(dim=1).item()
        #              argmax → (1,) → 取标量 0/1/2/3
    return LABELS[pred]  # → "伤感"/"深情"/"释怀"/"温暖"


# ===================== 8. 运行 =====================
if __name__ == "__main__":
    model, tokenizer = train()

    print("\n=== 测试 ===")
    for t in [
        "雪下得那么深下得那么认真",
        "我想和你在一起哪怕一天也可以",
        "不找了不再找了",
    ]:
        print(f"  {predict(model, tokenizer, t):4s} ← {t}")
