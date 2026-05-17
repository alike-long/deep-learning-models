import math
import pandas as pd
import torch as t
from torch import nn
from d2l import torch as d2l
from text10086 import PositionalEncoding


class FFn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        return x
lffn=FFn(4,4,8)
x=t.ones(4)
y=lffn(x)
print(y.shape)
class AddNorm(nn.Module):
    def __init__(self,norm_shape,dropout_rate):
        super(AddNorm, self).__init__()
        self.dropout=nn.Dropout(dropout_rate)
        self.ln=nn.LayerNorm(norm_shape)

    def forward(self, x,y):
        return self.ln(self.dropout(y)+x)
class Encoderblock(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout_rate,norm_shape,num_heads,num_layers,num_hiddens):
        super(Encoderblock,self).__init__()
        self.attention=d2l.MultiHeadAttention(num_hiddens,num_heads,dropout_rate)
        self.AddNorm1=AddNorm(norm_shape,dropout_rate)
        self.AddNorm2=AddNorm(norm_shape,dropout_rate)
        self.FFN=FFn(input_size,hidden_size,output_size)
    def forward(self,x,valid_lens):
        tx=x
        attnx=self.attention(x,x,x,valid_lens)
        x=self.AddNorm1(x,attnx)
        ffcx=self.FFN(x)
        x=self.AddNorm2(x,ffcx)
        return x

x=t.ones((2,10,4))
valid_lens=t.tensor([10,10])
eb=Encoderblock(4,4,4,0.5,[4],4,4,4)
y=eb(x,valid_lens)
print(y.shape)


class Decoderblock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate, num_heads, num_layers, num_hiddens, i):
        super(Decoderblock, self).__init__()
        self.i = i
        # ✅ 加 batch_first=True
        self.attention1 = nn.MultiheadAttention(num_hiddens, num_heads, dropout_rate, batch_first=True)
        self.ADDnorm1 = AddNorm(num_hiddens, dropout_rate)
        self.attention2 = nn.MultiheadAttention(num_hiddens, num_heads, dropout_rate, batch_first=True)
        self.AddNorm2 = AddNorm(num_hiddens, dropout_rate)
        self.FFN = FFn(input_size, hidden_size, output_size)
        self.AddNorm3 = AddNorm(num_hiddens, dropout_rate)

    def mask_attention(self, x):
        seq_len = x.size(1)
        mask = t.triu(t.ones(seq_len, seq_len, device=x.device), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, state):
        enc_outputs, enc_valid_lens, history = state

        if history[self.i] is None:
            key_values = x
        else:
            key_values = t.cat([history[self.i], x], dim=1)
        history[self.i] = key_values

        dec_valid_lens = self.mask_attention(x)

        # ✅ 官方 attention 必须接收两个输出！
        attn1, _ = self.attention1(x, key_values, key_values, attn_mask=dec_valid_lens)
        x = self.ADDnorm1(x, attn1)

        attn2, _ = self.attention2(x, enc_outputs, enc_outputs)
        x = self.AddNorm2(x, attn2)

        ffcx = self.FFN(x)
        x = self.AddNorm3(x, ffcx)
        return x, state



class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,num_heads,num_layers,num_hiddens,dropout_rate):
        super(TransformerEncoder,self).__init__()
        self.vocab_size=vocab_size
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(vocab_size,hidden_size)
        self.pos_encoding=PositionalEncoding(hidden_size)
        self.layer= nn.Sequential(*[
            Encoderblock(hidden_size,num_heads,dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self,x,valid_lens):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layer:
            x = layer(x,valid_lens)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,hidden_size,num_heads,num_layers,num_hiddens,dropout_rate):
        super(TransformerDecoder,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.embedding=nn.Embedding(vocab_size,hidden_size)
        self.pos_encoding=PositionalEncoding(hidden_size)
        self.layer= nn.Sequential(*[
            Decoderblock(hidden_size,num_heads,dropout_rate,i)
            for i in range(num_layers)
        ])
        self.fc =nn.Linear(hidden_size,vocab_size)

    def forward(self,x,state):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layer:
            x = layer(x,state)
        retuen = self.fc(x),state
class TransformerModel(nn.Module):
    def __init__(self,vocab_size,hidden_size,num_heads,num_layers,num_hiddens,dropout_rate):
        super(TransformerModel,self).__init__()
        self.vocab_size=vocab_size
        self.encoder=TransformerEncoder(vocab_size,hidden_size,num_heads,num_layers,dropout_rate)
        self.decoder=TransformerDecoder(vocab_size,hidden_size,num_heads,num_layers,dropout_rate)

    def forward(self,x,state):
        x=self.encoder(x,state)
        state=self.decoder(x,state)
        x=self.decoder(x,state)
        return x
net=TransformerModel(28,28,4,4,4,4)
print(net)







