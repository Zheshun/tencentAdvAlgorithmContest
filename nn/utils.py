# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from gensim.models import word2vec
import pandas as pd


MAX_VOCAB_SIZE = 300000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def builde_embedding_pretrained(vocab_path, dataset, col):
    model = word2vec.Word2Vec.load(dataset + '/w2v_0529/' + col + '_w2vmodel_0529')  # 预训练词向量
    vocab = pkl.load(open(vocab_path, 'rb'))
    matrix = []
    model_size = model.vector_size
    for word, index in vocab.items():
        ind_emb = [index]
        try:
            ind_emb.extend(list(model[word]))
        except:
            # if '男' == word:         # 加入性别特征
            #     ind_emb.extend(list(np.ones(model_size)))
            # elif '女' == word:
            #     ind_emb.extend(list(np.zeros(model_size)))
            # else:
            ind_emb.extend(list(np.random.rand(model_size)))
        matrix.append(ind_emb)
    ind_emb_pandas = pd.DataFrame(matrix)
    ind_emb_pandas = ind_emb_pandas.sort_values(0)  # 第1列是index
    ind_emb_pandas = ind_emb_pandas.iloc[:, 1:]
    return ind_emb_pandas


def build_dataset(dataset, config, ues_word, tag):
    vocabs, train, dev, test = [], [], [], []
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    for col in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry', 'product_category', 'time']: #, 'click_times']:
        if os.path.exists(dataset + '/data/' + col + '_' + tag + '_vocab.pkl'):
            vocab = pkl.load(open(dataset + '/data/' + col + '_' + tag + '_vocab.pkl', 'rb'))
        else:
            vocab = build_vocab(dataset + '/data/' + col + '_' + tag + '_train.txt', tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(vocab, open(dataset + '/data/' + col + '_' + tag + '_vocab.pkl', 'wb'))
        print(f"Vocab size: {len(vocab)}")
        vocabs.append(vocab)

        embeddings = builde_embedding_pretrained(dataset + '/data/' + col + '_' + tag + '_vocab.pkl', dataset, col)
        embedding_pretrained = torch.tensor(embeddings.values.astype('float32'))
        config.embedding_pretrained.append(embedding_pretrained)
        config.embed += embedding_pretrained.size(1)    # 第二种方式 config.embed.append(embedding_pretrained.size(1))

        def load_dataset(path, pad_size=32):
            contents = []
            with open(path, 'r', encoding='UTF-8') as f:
                for line in tqdm(f):
                    lin = line.strip()
                    if not lin:
                        continue
                    content, label = lin.split('\t')
                    words_line = []
                    token = tokenizer(content)
                    seq_len = len(token)
                    if pad_size:
                        if len(token) < pad_size:
                            token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                        else:
                            token = token[:pad_size]
                            seq_len = pad_size
                    # word to id
                    for word in token:
                        words_line.append(vocab.get(word, vocab.get(UNK)))
                    contents.append((words_line, int(label), seq_len))
            return contents  # [([...], 0, 23), ([...], 1, 11), ...]
        train.append(load_dataset(dataset + '/data/' + col + '_' + tag + '_train.txt', config.pad_size))
        dev.append(load_dataset(dataset + '/data/' + col + '_' + tag + '_dev.txt', config.pad_size))
        test.append(load_dataset(dataset + '/data/' + col + '_' + tag + '_test.txt', config.pad_size))
    return vocabs, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches[0]) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches[0]) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] - 1 for _ in datas]).to(self.device) # 因为从1-10不是0-9

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = [_[self.index * self.batch_size: len(self.batches[0])] for _ in self.batches]
            self.index += 1
            batches = [self._to_tensor(_) for _ in batches]
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = [_[self.index * self.batch_size: (self.index + 1) * self.batch_size] for _ in self.batches]
            self.index += 1
            batches = [self._to_tensor(_) for _ in batches]
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    pass
