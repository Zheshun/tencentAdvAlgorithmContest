# -*- coding: utf-8 -*-
import os
import pandas as pd
from gensim.models import word2vec
import pickle as pkl

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
PROJECT_PATH = os.path.abspath(CURRENT_PATH + '/../')


def make_w2v(size=128, min_count=2, word='creative_id', model_save_name='creativeid128_w2vmodel_0517'):
    # 构造原始数据集 [['ad_id1', 'ad_id2'....]]
    print('构造原始数据集')
    train_user_click_ad_y = pkl.load(open(os.path.abspath(PROJECT_PATH + 'data/train_user_click_ad.pkl'), 'rb'))
    train_user_click_ad = train_user_click_ad_y.drop(['age', 'gender'], axis=1)

    test_click_log = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/test/click_log.csv'))
    test_ad = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/test/ad.csv'))
    test_user_click_ad = pd.merge(test_click_log, test_ad, on='creative_id')
    # 合并训练集和测试集
    print('合并训练集和测试集')
    train_test_user_click_ad = pd.concat([train_user_click_ad, test_user_click_ad])
    # 按时间排序
    train_test_user_click_ad_timesort = train_test_user_click_ad.sort_values(by='time')
    def add_list(data):
        res = []
        for each in data:
            each = str(each)
            res.append(each)
        return res
    usr_creative_id_list = train_test_user_click_ad_timesort.groupby('user_id').aggregate({word: add_list}).reset_index()
    # 去除异常用户
    # print('去除前 usr_creative_id_list 的len：', len(usr_creative_id_list))
    # train_user_clean = pkl.load(open('data/followyu/0524/train_user_clean.pkl', 'rb'))
    # test_user_clean = pkl.load(open('data/followyu/0524/test_user_clean.pkl', 'rb'))
    # train_test_user_clean = train_user_clean + test_user_clean
    # usr_creative_id_list = usr_creative_id_list[usr_creative_id_list['user_id'].isin(tuple(train_test_user_clean))]
    # print('去除后 usr_creative_id_list 的len：', len(usr_creative_id_list))
    # 训练 w2v 并保存
    print('开始训练 w2v')
    model = word2vec.Word2Vec(list(usr_creative_id_list[word].values), sg=1, min_count=min_count, window=100, size=size, iter=20, workers=16)
    model.save(os.path.abspath(PROJECT_PATH + '/data/0525_8/w2v_0529/' + model_save_name))


if __name__ == '__main__':
    for col in ['creative_id', 'ad_id', 'product_id', 'advertiser_id']:
        make_w2v(128, 2, col, col + '_w2vmodel_0529')
    for col in ['industry', 'product_category', 'time', 'click_times']:
        make_w2v(10, 1, col, col + '_w2vmodel_0529')