"""
BERT 微调 — 任务2：实体标注（token 级）
========================================
这是 BERT 另一种微调方式：
  - 输入一整句 → 每个 token 的输出都接 Linear → 每个 token 一个标签
  - 跟任务1 的区别：不用 [CLS] 的 pooled，用全部 token 的输出

任务：标注歌词中每个字属于哪种"情感实体"
  O    = 普通字
  B-SAD = 伤感词开头
  I-SAD = 伤感词中间
  B-WARM= 温暖词开头
  I-WARM= 温暖词中间
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===================== 0. 导入 BERT =====================
from bert_model import BERTModel

# ===================== 1. 标签体系 =====================
# BIO 标注：B=开头, I=中间, O=不是实体
ID2TAG = ["O", "B-SAD", "I-SAD", "B-WARM", "I-WARM"]
TAG2ID = {t: i for i, t in enumerate(ID2TAG)}
PAD_TAG_ID = -100        # padding 位置的标签，CrossEntropyLoss 会忽略它


# ===================== 2. 造标注数据 =====================
# 每行：(原始句子, 每个字的标签)
# 标签字符对应：O=O, S=B-SAD, s=I-SAD, W=B-WARM, w=I-WARM
raw_data = [
    ("雪   下   得   那   么   深   下   得   那   么   认   真",
     "O   O   O   O   O   O   O   O   O   O   O   O"),

    ("爱   得   那   么   认   真   可   还   是   听   见   了   不   可   能",
     "O   O   O   O   O   O   O   O   O   O   O   O   O   O   O"),

    ("我   想   和   你   在   一   起",
     "O   W   w   w   w   w   w"),

    ("悲   伤   的   歌   越   唱   越   难   过",
     "S   s   O   O   O   O   O   S   s"),

    ("你   的   笑   容   那   么   温   柔",
     "O   O   W   w   w   w   W   w"),

    ("找   不   到   了   你   给   过   我   的   温   柔",
     "S   s   s   O   O   O   O   O   O   O   W   w"),

    ("我   陪   你   流   浪   去   远   方",
     "O   O   O   O   O   O   W   w"),

    ("失   去   你   以   后   我   才   懂   了   孤   单",
     "S   s   s   O   O   O   O   O   O   O   S   s"),

    ("世   界   很   暗   但   是   有   你   在   就   亮   了",
     "O   O   O   O   O   O   O   O   O   O   W   w   w"),

    ("说   散   你   想   很   久   了   吧   我   不   想   拆   穿   你",
     "O   O   O   O   O   O   O   O   O   O   O   O   O   O   O"),

    ("不   找   了   不   再   找   了   就   让   过   去   随   风   飘   散",
     "O   O   O   O   O   O   O   O   O   O   O   O   O   O   O   O"),

    ("我   回   不   到   童   年   了   但   我   还   能   感   受   温   暖",
     "O   O   O   O   O   O   O   O   O   O   O   O   O   O   W   w"),

    ("雪   中   的   伤   痕   深   深   印   在   心   里",
     "O   O   O   S   s   O   O   O   O   O   O"),

    ("我   想   带   你   去   海   边   看   日   落",
     "O   O   O   O   O   W   w   w   w   w   w"),

    ("你   像   风   一   样   吹   完   就   散",
     "O   O   O   O   O   O   O   O   O   O"),

    ("爱   与   被   爱   都   不   是   简   单   的   事",
     "O   O   O   O   O   O   O   O   O   O   O"),

    ("你   离   开   以   后   我   的   世   界   只   剩   黑   暗",
     "O   O   O   O   O   O   O   O   O   O   O   S   s"),

    ("温   暖   的   阳   光   洒   在   你   脸   庞",
     "W   w   O   O   O   O   O   O   O   O"),

    ("那   些   甜   蜜   的   回   忆   都   是   我   最   珍   贵   的   宝   贝",
     "O   O   W   w   O   O   O   O   O   O   O   O   O   O   O   O"),

    ("我   在   人   海   中   寻   找   遗   失   的   你",
     "O   O   O   O   O   O   O   S   s   s   O"),
]

# 把原始标注转成 (句子无空格, 标签列表)
annotated = []
for text_spaced, tag_spaced in raw_data:
    chars = text_spaced.split()
    tags = tag_spaced.split()
    tag_ids = [TAG2ID[t] for t in tags]
    sentence = "".join(chars)
    annotated.append((sentence, tag_ids))


# ===================== 3. 简易分词器 =====================
class SimpleTokenizer:
    def __init__(self, sentences):
        all_chars = sorted(set("".join(sentences)))
        self.vocab = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"] + all_chars
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.vocab_size = len(self.vocab)

        # 特殊 token 的 id
        self.pad_id = self.c2i["[PAD]"]
        self.cls_id = self.c2i["[CLS]"]
        self.sep_id = self.c2i["[SEP]"]

    def encode(self, text, max_len=50):
        """[CLS] + 每个字 + [SEP] + [PAD]"""
        ids = [self.cls_id]
        for c in text[:max_len - 2]:
            ids.append(self.c2i.get(c, self.c2i["[UNK]"]))
        ids.append(self.sep_id)
        ids += [self.pad_id] * (max_len - len(ids))
        return ids[:max_len]


# ===================== 4. Dataset =====================
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=50):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tag_ids = self.data[idx]

        # 编码输入
        input_ids = self.tok.encode(sentence, self.max_len)
        # 对齐标签：[CLS]→忽略, 每个字→对应标签, [SEP]→忽略, [PAD]→忽略
        label_ids = [PAD_TAG_ID]   # [CLS] 不算
        for t in tag_ids[:self.max_len - 2]:
            label_ids.append(t)
        label_ids.append(PAD_TAG_ID)   # [SEP] 不算
        label_ids += [PAD_TAG_ID] * (self.max_len - len(label_ids))
        label_ids = label_ids[:self.max_len]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        mask = (input_ids == self.tok.pad_id)
        return input_ids, mask, label_ids


# ===================== 5. 标注模型 =====================
# ★ 关键：token级任务用全部输出 (batch, seq, d_model)，不是 pooled
class BERTForNER(nn.Module):
    def __init__(self, vocab_size, num_tags=5):
        super().__init__()
        self.bert = BERTModel(vocab_size, d_model=128, n_heads=4,
                              n_layers=3, d_ff=512, max_len=50, dropout=0.1)
        # 每个 token 输出 128 维 → 映射到 5 种标签
        self.classifier = nn.Linear(128, num_tags)

    def forward(self, input_ids, key_padding_mask=None):
        x, _ = self.bert(input_ids, key_padding_mask=key_padding_mask)
        # x: (batch, seq_len, d_model)  ← 每个 token 都有输出
        logits = self.classifier(x)     # (batch, seq_len, num_tags)
        return logits


# ===================== 6. 训练 =====================
def train():
    sentences = [s for s, _ in annotated]
    tokenizer = SimpleTokenizer(sentences)
    dataset = NERDataset(annotated, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = BERTForNER(tokenizer.vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TAG_ID)
    # ↑ ignore_index=-100: CrossEntropyLoss 遇到 -100 直接跳过，不参与损失计算
    #   这样 [CLS] [SEP] [PAD] 位置都不会影响训练

    print(f"词表: {tokenizer.vocab_size} | 样本: {len(dataset)}")
    for epoch in range(100):
        total_loss = 0
        for input_ids, mask, labels in loader:
            opt.zero_grad()
            logits = model(input_ids, key_padding_mask=mask)  # (4, seq, 5)
            # loss_fn 期望: (N, C) 和 (N,)
            loss = loss_fn(logits.reshape(-1, 5), labels.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:2d} | loss:{total_loss:.2f}")

    return model, tokenizer


# ===================== 7. 预测并可视化 =====================
def predict_and_show(model, tokenizer, text):
    model.eval()
    ids = torch.tensor([tokenizer.encode(text)]).long()
    mask = (ids == tokenizer.pad_id)
    with torch.no_grad():
        logits = model(ids, key_padding_mask=mask)   # (1, seq, 5)
        preds = logits.argmax(dim=-1).squeeze(0)      # (seq,)

    # 逐字打印
    result = []
    for i, c in enumerate(text):
        pid = preds[i + 1].item()   # +1 跳过 [CLS]
        tag = ID2TAG[pid] if pid >= 0 else "?"
        result.append(f"{c}({tag})")
    print("  " + " ".join(result))
    model.train()


# ===================== 8. 运行 =====================
if __name__ == "__main__":
    model, tokenizer = train()

    print("\n=== 测试标注 ===")
    tests = [
        "悲伤的歌越唱越难过",
        "你的笑容那么温柔",
        "找不到你给过的温柔",
        "世界很暗但是有你在我身边",
        "失去你以后我才懂了孤单",
    ]
    for t in tests:
        print(f"输入: {t}")
        predict_and_show(model, tokenizer, t)
        print()
