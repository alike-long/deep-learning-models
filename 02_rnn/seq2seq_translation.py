import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F  # 这里修复了

# ===================== 1. 工具函数 & 屏蔽损失 =====================
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        loss = super().forward(pred.permute(0, 2, 1), label)
        return (loss * weights).mean(dim=1)

# ===================== 2. Encoder / Decoder / Seq2Seq =====================
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.gru(X)
        return output, state

# ------------------- 加性注意力 -------------------
class Attention(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.W_q = nn.Linear(num_hiddens, num_hiddens)
        self.W_k = nn.Linear(num_hiddens, num_hiddens)
        self.w = nn.Linear(num_hiddens, 1)

    def forward(self, query, keys_values):
        q = self.W_q(query).unsqueeze(1)
        k = self.W_k(keys_values).transpose(0, 1)
        scores = self.w(torch.tanh(q + k)).squeeze(-1)
        attn_weights = F.softmax(scores, dim=1)
        values = keys_values.transpose(0, 1)
        output = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)
        return output

# ------------------- 带注意力的 Decoder -------------------
class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(num_hiddens)
        self.gru = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        enc_all_outputs, hidden_state = enc_outputs
        return enc_all_outputs, hidden_state  # 保存编码器全部输出

    def forward(self, X, state):
        enc_all_outputs, hidden_state = state
        X = self.embedding(X).permute(1, 0, 2)

        # 核心：用解码器当前隐藏态 Q 去关注编码器全部输出 KV
        attn_output = self.attention(hidden_state[-1], enc_all_outputs)

        context = attn_output.unsqueeze(0).repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)

        output, state = self.gru(X_and_context, hidden_state)
        output = self.dense(output).permute(1, 0, 2)
        return output, (enc_all_outputs, state)

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.encoder = Seq2SeqEncoder(src_vocab_size, embed_size, num_hiddens, num_layers, dropout)
        self.decoder = Seq2SeqDecoder(tgt_vocab_size, embed_size, num_hiddens, num_layers, dropout)

    def forward(self, src_X, tgt_X):
        enc_out = self.encoder(src_X)
        dec_state = self.decoder.init_state(enc_out)
        tgt_hat, _ = self.decoder(tgt_X, dec_state)
        return tgt_hat

# ===================== 3. 数据集 =====================
cn_sentences = [
    "我 爱 你", "你 好", "今 天 很 好", "我 在 学 习",
    "明 天 会 下 雨", "我 喜 欢 机 器 学 习", "你 很 棒", "加 油"
]
en_sentences = [
    "i love you", "hello", "today is good", "i am studying",
    "it will rain tomorrow", "i like machine learning", "you are great", "come on"
]

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

def build_vocab(sent_list):
    tokens = [PAD, BOS, EOS]
    for sent in sent_list:
        tokens.extend(sent.split())
    vocab = {w:i for i,w in enumerate(sorted(list(set(tokens))))}
    return vocab

src_vocab = build_vocab(cn_sentences)
tgt_vocab = build_vocab(en_sentences)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

def sent2ids(sent, vocab):
    return [vocab[w] for w in sent.split()] + [vocab[EOS]]

def pad_seq(seq, max_len, pad_id):
    if len(seq) < max_len:
        seq += [pad_id]*(max_len - len(seq))
    return seq[:max_len]

class ZHENDataset(Dataset):
    def __init__(self, cn_list, en_list, src_vocab, tgt_vocab, max_len=12):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.pad_s = src_vocab[PAD]
        self.pad_t = tgt_vocab[PAD]
        self.data = []
        for cn, en in zip(cn_list, en_list):
            src_ids = sent2ids(cn, src_vocab)
            tgt_ids = sent2ids(en, tgt_vocab)
            src_ids = pad_seq(src_ids, max_len, self.pad_s)
            tgt_ids = pad_seq(tgt_ids, max_len, self.pad_t)
            self.data.append((src_ids, tgt_ids))

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        src_len = len([x for x in src_ids if x != self.pad_s])
        tgt_len = len([x for x in tgt_ids if x != self.pad_t])
        return (torch.tensor(src_ids), torch.tensor(src_len),
                torch.tensor(tgt_ids), torch.tensor(tgt_len))

    def __len__(self):
        return len(self.data)

# ===================== 4. 训练 & 预测 =====================
def train_seq2seq(net, loader, lr, num_epochs, device):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = MaskedSoftmaxCELoss()
    bos_id = tgt_vocab[BOS]

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        total_tokens = 0
        for src_X, src_len, tgt_X, tgt_len in loader:
            src_X, tgt_X = src_X.to(device), tgt_X.to(device)
            tgt_len = tgt_len.to(device)
            bos = torch.full((src_X.shape[0],1), bos_id, device=device)
            dec_in = torch.cat([bos, tgt_X[:,:-1]], dim=1)

            pred = net(src_X, dec_in)
            loss = loss_fn(pred, tgt_X, tgt_len)

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            total_loss += loss.sum().item()
            total_tokens += tgt_len.sum().item()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss/total_tokens:.3f}")

def predict_seq2seq(net, cn_sent, src_vocab, tgt_vocab, max_len, device):
    net.eval()
    eos_id = tgt_vocab[EOS]
    bos_id = tgt_vocab[BOS]
    pad_id = src_vocab[PAD]

    src_ids = pad_seq(sent2ids(cn_sent, src_vocab), max_len, pad_id)
    src_X = torch.tensor([src_ids], device=device)

    with torch.no_grad():
        enc_out = net.encoder(src_X)
        dec_state = net.decoder.init_state(enc_out)
        dec_X = torch.tensor([[bos_id]], device=device)
        pred_ids = []
        for _ in range(max_len):
            Y, dec_state = net.decoder(dec_X, dec_state)
            dec_X = Y.argmax(2)
            p = dec_X.item()
            if p == eos_id: break
            pred_ids.append(p)

    id2w = {v:k for k,v in tgt_vocab.items()}
    return " ".join([id2w[i] for i in pred_ids])

# ===================== 5. 运行 =====================
if __name__ == "__main__":
    embed_size = 64
    num_hiddens = 128
    num_layers = 1
    batch_size = 2
    max_len = 12
    lr = 0.001
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ZHENDataset(cn_sentences, en_sentences, src_vocab, tgt_vocab, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = Seq2Seq(src_vocab_size, tgt_vocab_size, embed_size, num_hiddens, num_layers)
    train_seq2seq(net, dataloader, lr, epochs, device)

    test_cn = "我 爱 我"
    res = predict_seq2seq(net, test_cn, src_vocab, tgt_vocab, max_len, device)
    print("\n输入：", test_cn)
    print("输出：", res)