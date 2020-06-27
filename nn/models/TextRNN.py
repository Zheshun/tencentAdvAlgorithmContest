# coding: UTF-8
import torch
import torch.nn as nn


class Config(object):

    """配置参数"""
    def __init__(self, dataset, tag):
        self.model_name = 'TextRNN'
        self.dataset = dataset
        self.tag = tag
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        # embeddings = builde_embedding_pretrained(self.vocab_path)
        self.embedding_pretrained = [] #torch.tensor(embeddings.values.astype('float32'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 5000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 10 if tag == 'age' else 2                    # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 256                                           # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = 0
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数


'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = [nn.Embedding.from_pretrained(_, freeze=False) for _ in config.embedding_pretrained]
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)  # 第二种方式 [for _ in config.embed]
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        # 第二种方式 self.fc_f = nn.Linear(config.num_classes * len(config.embed), config.num_classes)

    def forward(self, x__):
        for i, (x_, y) in enumerate(x__):
            x, _ = x_
            self.embedding[i] = self.embedding[i].to('cuda')
            out_i = self.embedding[i](x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
            if i == 0:
                out = out_i
            else:
                out = torch.cat([out, out_i], dim=2)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    def forward_2(self, x__):
        for i, (x_, y) in enumerate(x__):
            x, _ = x_
            out = self.embedding[i](x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
            out, _ = self.lstm[i](out)
            out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
            if i == 0:
                out_f = out
            else:
                out_f = torch.cat([out_f, out], dim=1)
        out = self.fc_f(out_f)
        return out
