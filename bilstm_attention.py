import chainer
# coding: utf-8
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import numpy as np

class LSTM(Chain):
    def __init__(self, n_mid_units=128, n_out=4, n_vocab=100001, n_embed_units=128, dropout=0.25, initialW=None):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab,
                                   n_embed_units,
                                   initialW=initialW,
                                   ignore_label=-1)
            self.lstm=L.NStepBiLSTM(n_layers=1, in_size=n_embed_units, out_size=n_mid_units, dropout=dropout) 
            self.l_attn = L.Linear(n_mid_units*2, 1)
            self.l3 = L.Linear(n_mid_units*2, n_out)
            
    def __call__(self, xs, batchsize):
        doc_len = xs[0].shape[0]
        xs_embed = [F.dropout(self.embed(Variable(item)), ratio=0.5) for item in xs]
        hy, cy, ys = self.lstm(hx=None, cx=None, xs=xs_embed)
        ys = [F.dropout(item, ratio=0.25) for item in ys]

        # attentionの計算
        concat_ys = F.concat(ys, axis=0)
        attn = F.dropout(self.l_attn(concat_ys), ratio=0.25)
        split_attention = F.split_axis(attn, np.cumsum([len(item) for item in xs])[:-1], axis=0)
        split_attention_pad = F.pad_sequence(split_attention, padding=-1024.0)
        attn_softmax = F.softmax(split_attention_pad, axis=1)
        ys_pad = F.pad_sequence(ys, length=None, padding=0.0)
        ys_pad_reshape = F.reshape(ys_pad, (-1, ys_pad.shape[-1]))
        attn_softmax_reshape = F.broadcast_to(F.reshape(attn_softmax, (-1, attn_softmax.shape[-1])), ys_pad_reshape.shape)
        attention_hidden = ys_pad_reshape * attn_softmax_reshape
        attention_hidden_reshape = F.reshape(attention_hidden, (batchsize, -1, attention_hidden.shape[-1]))
        result = F.sum(attention_hidden_reshape, axis=1)
        
        y = self.l3(result)
        return y, attn_softmax[0].data
        
        
