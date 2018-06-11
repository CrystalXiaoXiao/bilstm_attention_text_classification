# coding: utf-8
from argparse import ArgumentParser

from collections import defaultdict
import six
import sys

import gensim
import gzip
import json
import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, iterators
from chainer import Link, Chain, ChainList
from chainer import report, training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from bilstm_attention import LSTM

from sklearn.metrics import f1_score
from chainer.cuda import to_cpu

def load_data(path, dic, n=None, lim=None):
    with open(path, "rt") as f:
        data = f.readlines()
    if n is not None:
        data = data[:n]
        
    target = []
    source = []
    for j, datum in enumerate(data):
        doc = datum[2:].split(" ")
        document_vec = []
        
        if (lim is not None) and (len(doc) >= lim):
            doc = doc[:lim]
        for word in doc:
            try:
                document_vec.append(dic[word])
            except KeyError:
                document_vec.append(dic['<UNK>'])
  
        source.append(document_vec)
        target.append(int(datum[0]))

    return np.array(source), np.array(target)
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--gpu", default=-1, type=int)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--doc_len", default=350, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--embedding", default='w2v.midasi.256.100K.bin',type=str)
    parser.add_argument("--train", default="train.csv" type=str)
    parser.add_argument("--test", default="test.csv" , type=str)
    parser.add_argument("--nunits", default=128, type=int)
    args = parser.parse_args()

    model_w2v = gensim.models.KeyedVectors.load_word2vec_format(args.embedding, binary=True)
    word2index = {w: i for i, w in enumerate(model_w2v.index2word)}
    
    ## dataset
    doc_len = args.doc_len

    train_x, train_y = load_data(args.train, word2index, lim=doc_len)
    test_x, test_y = load_data(args.test, word2index, lim=doc_len)

    batchsize = args.batchsize
    
    n_vocab, n_embed_units = model_w2v.vectors.shape

    # 隠れ層のユニット数
    n_mid_units = args.nunits

    #モデルの定義
    model = LSTM(n_mid_units=n_mid_units,
                 n_vocab=n_vocab,
                 n_embed_units=n_embed_units,
                 initialW=model_w2v.vectors)
    model.embed.disable_update()

    #GPUを使うかどうか
    gpu_id = args.gpu
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(gpu_id).use()
        xp = cuda.cupy
        model.to_gpu()
    else:
        xp = np
        
    # Setup optimizer
    optimizer = optimizers.MomentumSGD()
    #optimizer = optimizers.Adam()
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
    #optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))
    max_epoch = args.epoch
    N = len(train_x)
    N_test = len(test_x)
    max_macro_f = 0
    for epoch in range(1, max_epoch + 1):        
        print('epoch', epoch, '/', max_epoch)
       
        # training        
        perm = np.random.permutation(N) 
        loss_train_list     = []
        accuracy_train_list = []
        for i in range(0, N, batchsize):

            x = [xp.array(doc, dtype=xp.int32) for doc in train_x[perm[i: i+batchsize]]]
            t = xp.array(train_y[perm[i: i+batchsize]], dtype=xp.int32)
            
            y, _ = model(x, len(perm[i:i + batchsize]))
            loss_train = F.softmax_cross_entropy(y, t)
            accuracy_train = F.accuracy(y, t)
            model.cleargrads()
            loss_train.backward()
            optimizer.update()
            
            loss_train_list.append(float(loss_train.data))
            accuracy_train_list.append(float(accuracy_train.data))
            
        print('train mean loss={}, accuracy={}'.format(np.mean(loss_train_list), np.mean(accuracy_train_list))) 

        # evaluation
        x = [xp.array(doc, dtype=xp.int32) for doc in test_x]
        t = xp.array(test_y, dtype=xp.int32)
            
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False): 
            y, _ = model(x, N_test)
        loss_test = F.softmax_cross_entropy(y, t)
        y = [np.argmax(item) for item in to_cpu(y.data)]
        macro_f = f1_score(to_cpu(t), np.array(y), average='macro')  
        print('test mean loss={}, f_score={}'.format(loss_test.data, macro_f))
        if macro_f > max_macro_f:
            max_macro_f = macro_f
            serializers.save_npz("bi-lstm_attention_best.model".format(str(max_macro_f)), model)        
        
        sys.stdout.flush()

if __name__ == "__main__":
        main()
        

