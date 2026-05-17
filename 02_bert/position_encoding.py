"""
BERT 预处理核心：三层编码 + MLM 掩码
=====================================
这是 BERT 最核心的预处理逻辑，包括两个部分：
  1. 三层编码：token + 位置 + 分段 → 相加 → LayerNorm → Dropout
  2. MLM 掩码：15% 的词做手脚 → 80%[MASK] / 10%随机 / 10%不变
"""
import torch
import torch.nn as nn
import random

# ================================================================
#  第一部分：三层编码 — BERTEmbedding
# ================================================================
# 为什么是三层而不是一层？
#   一层(token)只告诉模型"这个词是什么"
#   两层(+pos) 告诉模型"这个词在第几位"     ← Transformer 没有递归，不标位置分不清顺序
#   三层(+seg) 告诉模型"这个词属于哪句话"  ← BERT 输入是两句拼接的

class BERTEmbedding(nn.Module):
    """
    三层嵌入相加：
      x = token_embed(token) + pos_embed(位置) + seg_embed(属于哪句)
    然后过 LayerNorm + Dropout
    """
    def __init__(self, vocab_size=300, d_model=128, max_len=128, type_vocab_size=2, dropout=0.1):
        super().__init__()
        # ① token 嵌入：每个词 → 128维语义向量
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # ② 位置嵌入：第0位~第127位，每位置一个128维向量
        self.pos_embed = nn.Embedding(max_len, d_model)
        # ③ 分段嵌入：A句还是B句（只2种: 0或1）
        self.seg_embed = nn.Embedding(type_vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, seg_ids=None):
        # input_ids: (2, 16)  ← batch=2, seq_len=16
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)

        # 位置编号: [[0,1,2,...,15], [0,1,2,...,15]]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        #   torch.arange(16) → (16,)
        #   .unsqueeze(0)    → (1, 16)
        #   .expand(2, -1)   → (2, 16)

        # ★ 三层相加（核心！）
        # ① token_embed: 每个id查表取128维
        x = self.token_embed(input_ids)
        #   input_ids:    (2, 16)        ← 16个id
        #   查表(17×128): 每个id→128维
        #   x:           (2, 16, 128)    ← 每个字变128维向量

        # ② pos_embed: 每个位置查表取128维（同一位置不同句子共用）
        x = x + self.pos_embed(positions)
        #   positions: (2, 16)           ← 16个位置编号
        #   pos_embed: (128, 128) 表     ← 128个位置×128维
        #   查表后:    (2, 16, 128)      ← 每个位置128维向量
        #   相加后 x: (2, 16, 128)       ← 逐元素加，形状不变

        # ③ seg_embed: 每个位置查分段表（全是0或全是1）
        x = x + self.seg_embed(seg_ids)
        #   seg_ids:  (2, 16)           ← 每个位置 0或1
        #   seg_embed: (2, 128) 表      ← 2种分段×128维
        #   查表后:   (2, 16, 128)      ← 每个位置128维
        #   相加后 x: (2, 16, 128)      ← 三次相加，形状始终不变

        # 3个128维向量逐元素相加，不是拼接。结果还是 (2, 16, 128)
        x = self.norm(x)       # LayerNorm: (2,16,128)→(2,16,128)  归一化
        x = self.dropout(x)    # Dropout: 形状不变，随机置零
        return x               # (2, 16, 128)


# ================================================================
#  第二部分：MLM 掩码策略
# ================================================================
# 预训练时随机遮住 15% 的词让模型猜，但这 15% 分三种处理：
#
#   原文: 我 爱 你 像 风 一 样
#
#   选中的 15%（比如"爱"和"风"）:
#     → 80% 换成 [MASK]  : 我 [MASK] 你 像 [MASK] 一 样    ← 主力
#     → 10% 换成随机词    : 我 [MASK] 你 像  亮  一 样     ← 防依赖[MASK]
#     → 10% 保持原词      : 我 [MASK] 你 像  风  一 样     ← 弥合偏差

# --------------------- 演示：MLM 掩码生成 ---------------------
def create_mlm_mask(input_ids, vocab_size, mask_token_id=103, pad_token_id=0):
    """
    输入: input_ids — (batch, seq_len)
    输出: masked_ids — 掩码后的输入
          labels     — 只有被掩码的位置有真实标签，其余是 -100（忽略）
    """
    batch, seq_len = input_ids.shape
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)   # -100 被 CrossEntropyLoss 忽略

    for b in range(batch):
        for pos in range(seq_len):
            token = input_ids[b, pos].item()

            if token == pad_token_id:           # [PAD] 不掩码
                continue
            if token in [101, 102]:             # [CLS]/[SEP] 不掩码，它们是特殊标记
                continue

            if random.random() > 0.15:          # 85% 的词不动
                continue

            # 这个位置被选中了 → 存真实标签
            labels[b, pos] = token

            rand = random.random()
            if rand < 0.8:                      # 80% → 换成 [MASK]
                masked_ids[b, pos] = mask_token_id
            elif rand < 0.9:                    # 10% → 换成随机词
                masked_ids[b, pos] = random.randint(4, vocab_size - 1)
            else:                               # 10% → 保持原词
                pass

    return masked_ids, labels


# ================================================================
#  第三部分：完整演示
# ================================================================
if __name__ == "__main__":
    # ===== 1. 模拟一个简易词表 =====
    chars = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]",
             "我", "爱", "你", "像", "风", "一", "样", "笑", "真", "好", "看"]
    # ★ 先 set 去重再 enumerate，保证每个字有唯一索引
    unique_chars = sorted(set(chars), key=lambda c: chars.index(c))
    vocab = {c: i for i, c in enumerate(unique_chars)}
    vocab_size = len(vocab)
    PAD_ID  = vocab["[PAD]"]
    CLS_ID  = vocab["[CLS]"]
    SEP_ID  = vocab["[SEP]"]
    MASK_ID = vocab["[MASK]"]

    # ===== 2. 两条输入（每句最长 16）=====
    raw = [
        (["我", "爱", "你", "像", "风", "一", "样"],        # A句
         ["你", "笑", "真", "好", "看"]),                    # B句
        (["我", "爱", "你"],                                 # A句
         ["像", "风", "一", "样"]),                           # B句
    ]

    def encode_pair(a_words, b_words, max_len=16):
        """[CLS] + A句每个字 + [SEP] + B句每个字 + [SEP] + [PAD]补满"""
        # a_words: ["我","爱","你","像","风","一","样"]  ← 7个字
        # b_words: ["你","笑","真","好","看"]            ← 5个字
        ids = [CLS_ID]
        #    ids: [3]   ← [CLS]
        ids += [vocab.get(w, vocab["[UNK]"]) for w in a_words]
        #    ids: [3, 5,6,7,8,9,10,11]  ← 每个字变1个id, 7个id
        ids.append(SEP_ID)
        #    ids: [3, 5,6,7,8,9,10,11, 4]  ← 9个id
        seg_a = len(ids)   # seg_a=9  ← A句区域到此结束
        ids += [vocab.get(w, vocab["[UNK]"]) for w in b_words]
        #    ids: [..., 4, 12,13,14,15,16]  ← 5个字→5个id
        ids.append(SEP_ID)
        #    ids: [3,5,6,7,8,9,10,11,4, 12,13,14,15,16,4]  ← 15个id, B句结束
        seg_b = len(ids) - seg_a  # seg_b = 15-9 = 6  ← B句区域
        ids += [PAD_ID] * (max_len - len(ids))
        #    ids: [...,4, 0]  ← 补1个[PAD], 凑满16
        #    ids: (16,)  ← 最终固定长度
        seg = [0]*seg_a + [1]*seg_b   # [0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1] ← 15个
        seg += [0] * (max_len - len(seg))
        #    seg: [..., 0] ← 补1个0, 凑满16
        #    seg: (16,)
        return ids[:max_len], seg[:max_len]

    input_ids_list, seg_list = [], []
    for a, b in raw:
        ids, seg = encode_pair(a, b, max_len=16)
        input_ids_list.append(ids)
        seg_list.append(seg)

    # 2条句子 → 拼成 batch
    input_ids = torch.tensor(input_ids_list)   # input_ids_list有2个(16,) → 叠成 (2, 16)
    seg_ids = torch.tensor(seg_list)            # 同上: (2, 16)

    # ===== 3. 看三层编码的维度变化 =====
    emb = BERTEmbedding(vocab_size, d_model=128, max_len=128)
    x = emb(input_ids, seg_ids)
    print("=" * 50)
    print("【维度变化全流程】")
    print(f"\n  输入:")
    print(f"    input_ids: {input_ids.shape}     ← 2句话, 每句16个id")
    print(f"    seg_ids:   {seg_ids.shape}     ← 2句话, 每个位置标注A句(0)或B句(1)")
    print(f"\n  三层编码（每一步形状不变）:")
    print(f"    token_embed(id) → (2, 16, 128)  ← 每个字查表→128维")
    print(f"    + pos_embed(pos) → (2, 16, 128)  ← 每个位置查表→128维")
    print(f"    + seg_embed(seg) → (2, 16, 128)  ← A/B句查表→128维")
    print(f"    → LayerNorm     → (2, 16, 128)  ← 归一化")
    print(f"    → Dropout       → (2, 16, 128)  ← 随机置零")
    print(f"\n  输出 x: {tuple(x.shape)}")
    print(f"  含义: 16个字 × 128维 = 每个字的语义+位置+分段信息")

    # ===== 4. 看 MLM 掩码的维度变化 =====
    random.seed(42)
    masked_ids, labels = create_mlm_mask(input_ids, vocab_size, MASK_ID, PAD_ID)
    # masked_ids: (2, 16)  — 跟input_ids形状一样，但某些位置被改了
    # labels:     (2, 16)  — 只有被改的位置存真实id，其余是-100

    print("=" * 50)
    print("【MLM 掩码维度变化】")
    print(f"  input_ids:   (2, 16)    ← 原始输入")
    print(f"  masked_ids:  (2, 16)    ← 15%做手脚后")
    print(f"  labels:      (2, 16)    ← 被做手脚的位置存真实id, 其余-100")
    print(f"\n  训练时:")
    print(f"    把 masked_ids 送进 BERT → 输出 (2, 16, 128)")
    print(f"    过 MLM头 (128→词表大小)  → (2, 16, 词表大小)")
    print(f"    CrossEntropyLoss(ignore_index=-100):")
    print(f"      只算 labels≠-100 的位置的损失")
    print(f"      通常 16个位置里只有 1~2个参与计算")

    print("\n" + "=" * 50)
    print("【MLM 掩码演示】")
    id2c = {v: k for k, v in vocab.items()}
    for b in range(2):
        print(f"\n  原句: {' '.join(id2c[t.item()] for t in input_ids[b])}")
        print(f"  掩码: {' '.join(id2c[t.item()] for t in masked_ids[b])}")
        label_str = []
        for i, t in enumerate(labels[b]):
            if t.item() == -100:
                label_str.append("·")
            else:
                label_str.append(id2c[t.item()])
        print(f"  标签: {'  '.join(label_str)}")
        print(f"        ↑ '·'=不参与损失(-100)，有字=要还原的词")

    print("\n" + "=" * 50)
    print("【总结】")
    print("  三层编码: token + pos + seg → LayerNorm → Dropout")
    print("            (2,16) ×3个Embedding → (2,16,128) 形状始终不变")
    print("  MLM掩码: 15%选词 → 80%[MASK] / 10%随机 / 10%原词")
    print("  labelsssssss:   只有被做的位置有真实id，其余-100被loss忽略")
