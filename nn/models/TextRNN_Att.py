# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):

    """配置参数"""
    def __init__(self, dataset, tag):
        self.model_name = 'TextRNN_Att'
        self.dataset = dataset
        self.tag = tag
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 5000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 10 if tag == 'age' else 2                    # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 256                                          # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                      # 学习率
        self.embed = 0
        self.hidden_size = 512                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 256


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = [nn.Embedding.from_pretrained(_, freeze=False) for _ in config.embedding_pretrained]
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.Tensor(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x__):
        for i, (x_, y) in enumerate(x__):
            x, _ = x_
            self.embedding[i] = self.embedding[i].to('cuda')
            out_i = self.embedding[i](x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
            if i == 0:
                emb = out_i
            else:
                emb = torch.cat([emb, out_i], dim=2)
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
