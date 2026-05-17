import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    """BERT 的嵌入层：token + 位置 + 分段，最后加 LayerNorm + Dropout"""
    def __init__(self, vocab_size, d_model=256, max_len=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(type_vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, seg_ids=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embed(input_ids)
        x = x + self.pos_embed(positions)
        x = x + self.seg_embed(seg_ids)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class BERTAttention(nn.Module):
    """基于 nn.MultiheadAttention 的双向注意力"""
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, key_padding_mask=None):
        # key_padding_mask: (batch, seq_len), True 表示忽略该位置
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return out


class BERTFeedForward(nn.Module):
    """BERT 用 GELU，不是 ReLU"""
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))


class BERTLayer(nn.Module):
    """单层 BERT Encoder：Attention → FFN，Post-Norm + 残差"""
    def __init__(self, d_model=256, n_heads=8, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = BERTAttention(d_model, n_heads, dropout)
        self.ff = BERTFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = self.norm1(x + self.attn(x, key_padding_mask))
        x = self.norm2(x + self.ff(x))
        return x


class BERTPooler(nn.Module):
    """取 [CLS] 位置的输出，过一个 tanh 得到句向量"""
    def __init__(self, d_model=256):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.fc(x[:, 0]))


class BERTModel(nn.Module):
    """小 BERT：d_model=256, n_heads=8, n_layers=6, d_ff=1024"""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            BERTLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.pooler = BERTPooler(d_model)

    def forward(self, input_ids, seg_ids=None, key_padding_mask=None):
        x = self.embedding(input_ids, seg_ids)
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        pooled = self.pooler(x)
        return x, pooled


class MLMHead(nn.Module):
    """Masked Language Model 头：Linear → GELU → LayerNorm → Linear"""
    def __init__(self, d_model=256, vocab_size=30000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, x):
        return self.net(x)


class BERTForMLM(nn.Module):
    """小 BERT + MLM 头，用于预训练"""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.bert = BERTModel(vocab_size, d_model, n_heads, n_layers,
                              d_ff, max_len, dropout)
        self.mlm_head = MLMHead(d_model, vocab_size)

    def forward(self, input_ids, seg_ids=None, key_padding_mask=None):
        x, pooled = self.bert(input_ids, seg_ids, key_padding_mask)
        logits = self.mlm_head(x)
        return logits
