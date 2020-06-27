# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from gensim.models import word2vec
import pickle as pkl
from sklearn import model_selection
import lightgbm as lgb
from sklearn.metrics import accuracy_score

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
PROJECT_PATH = os.path.abspath(CURRENT_PATH + '/../')


def make_data_train(word, word2vec_model_name='creativeid300_w2vmodel_0518',w2v_size=300):
    print('开始load 训练集')
    train_user = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/train_preliminary/user.csv'))
    model = word2vec.Word2Vec.load(os.path.abspath(PROJECT_PATH + '/data/0520/' + word2vec_model_name))
    train_user_click_ad_y = pkl.load(open(os.path.abspath(PROJECT_PATH + 'data/train_user_click_ad.pkl'), 'rb'))

    print('开始emb')
    def add_list(data):
        res = []
        for each in data:
            each = str(each)
            res.append(each)
        return res
    usr_creative_id_list = train_user_click_ad_y.groupby('user_id').aggregate({word: add_list}).reset_index()
    def get_emb(ids):
        res = np.zeros(w2v_size)
        for id_ in ids:
            res += model[id_]
        return res / len(ids)    # avg
    usr_creative_id_list['word_id_emb'] = usr_creative_id_list[word].apply(get_emb)
    usr_creative_id_list = usr_creative_id_list.sort_values(by='user_id')
    creative_id_emb_expand = usr_creative_id_list['word_id_emb'].apply(pd.Series).reset_index()
    # usr_creative_id_emb_expand = pd.concat([usr_creative_id_list, creative_id_emb_expand], axis=1)
    creative_id_emb_expand['index'] = creative_id_emb_expand['index'].apply(lambda x: x + 1)
    usr_creative_id_emb_expand_y = pd.merge(creative_id_emb_expand, train_user, left_on='index', right_on='user_id')
    usr_creative_id_emb_expand_y = usr_creative_id_emb_expand_y.drop('index', axis=1)
    return usr_creative_id_emb_expand_y


def make_data_test(word2vec_model_name='creativeid300_w2vmodel_0518', w2v_size=300):
    print('开始load 测试集')
    model = word2vec.Word2Vec.load(os.path.abspath(PROJECT_PATH + '/data/' + word2vec_model_name))
    test_click_log = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/test/click_log.csv'))
    test_ad = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/test/ad.csv'))
    test_user_click_ad = pd.merge(test_click_log, test_ad, on='creative_id')

    print('开始emb')
    def add_list(data):
        res = []
        for each in data:
            each = str(each)
            res.append(each)
        return res
    usr_creative_id_list = test_user_click_ad.groupby('user_id').aggregate({'creative_id': add_list}).reset_index()

    def get_emb(ids):
        res = np.zeros(w2v_size)
        for id_ in ids:
            res += model[id_]
        return res / len(ids)  # avg
    usr_creative_id_list['creative_id_emb'] = usr_creative_id_list['creative_id'].apply(get_emb)
    usr_creative_id_list = usr_creative_id_list.sort_values(by='user_id')
    creative_id_emb_expand = usr_creative_id_list['creative_id_emb'].apply(pd.Series)

    return creative_id_emb_expand, usr_creative_id_list['user_id']


def allin_train_lgbm():
    train_user = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/train_preliminary/user.csv'))
    x_base = train_user.sort_values(by='user_id')
    for col in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']:
        usr_word_id_emb_expand_y = make_data_train(col, col + '_w2vmodel_0520', 128)
        x = usr_word_id_emb_expand_y.drop(['user_id', 'age', 'gender'], axis=1)
        # y_gender = usr_word_id_emb_expand_y['gender']
        # y_age = usr_word_id_emb_expand_y['age']
        x_base = pd.concat([x_base, x], axis=1)
    y_gender = x_base['gender']
    y_age = x_base['age']
    x = x_base.drop(['user_id', 'age', 'gender'], axis=1)
    print(x.shape)

    # creative_id_emb_expand, predict_user_id = make_data_test()
    # x_predict = creative_id_emb_expand

    # write_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/x.pkl'), x)
    # write_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/y_gender.pkl'), y_gender)
    # write_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/y_age.pkl'), y_age)
    # write_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/x_predict.pkl'), x_predict)
    # write_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/predict_user_id.pkl'), predict_user_id)

    # x = load_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/x.pkl'))
    # y_gender = load_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/y_gender.pkl'))
    # y_age = load_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/y_age.pkl'))
    # x_predict = load_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/x_predict.pkl'))
    # predict_user_id = load_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/predict_user_id.pkl'))
    #
    # feature_x_final = load_pickle(os.path.abspath(PROJECT_PATH + '/data/0519/feature_x_final.pkl'))
    # feature_x_final = pd.concat([feature_x_final, y_gender], axis=1)
    # x = pd.concat([feature_x_final, x], axis=1)
    # x = x.drop('user_id', axis=1)
    # x = x.replace('\\N', 0).astype(np.float)    # 填充均值

    print('y_gender train and predict')
    x_tran, x_test, y_tran, y_test = model_selection.train_test_split(x, y_age, test_size=0.2)

    for i in range(5):
        x_tran_per = x_tran.iloc[:, i*128:i*128+128]
        x_test_per = x_test.iloc[:, i*128:i*128+128]
        print(x_test_per.shape)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 11,
            'metric': 'multi_error',
            'num_leaves': 120,
            'min_data_in_leaf': 100,
            'learning_rate': 0.06,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.4,
            'lambda_l2': 0.5,
            'min_gain_to_split': 0.2,
            'verbose': -1,
        }
        print('Training...')
        trn_data = lgb.Dataset(x_tran_per, y_tran)
        val_data = lgb.Dataset(x_test_per, y_test)
        clf = lgb.train(params, trn_data, num_boost_round=1000, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
        y_tran_prob = clf.predict(x_tran_per, num_iteration=clf.best_iteration)
        y_test_prob = clf.predict(x_test_per, num_iteration=clf.best_iteration)
        if i == 0:
            y_tran_concat = pd.DataFrame(y_tran_prob)
            y_test_concat = pd.DataFrame(y_test_prob)
        else:
            y_tran_concat = pd.concat([y_tran_concat, pd.DataFrame(y_tran_prob)], axis=1)
            y_test_concat = pd.concat([y_test_concat, pd.DataFrame(y_test_prob)], axis=1)

    print('Training concat...')
    trn_data = lgb.Dataset(y_tran_concat, y_tran)
    val_data = lgb.Dataset(y_test_concat, y_test)
    clf = lgb.train(params, trn_data, num_boost_round=1000, valid_sets=[trn_data, val_data], verbose_eval=100,
                    early_stopping_rounds=100)
    # y_tran_prob = clf.predict(x_tran_per, num_iteration=clf.best_iteration)
    # y_test_prob = clf.predict(x_test_per, num_iteration=clf.best_iteration)
    #
    #
    # print('Predicting...')
    #
    # y_prob = clf.predict(x_predict, num_iteration=clf.best_iteration)
    # y_age_predict = [list(x).index(max(x)) for x in y_prob]

    # print("ACC score: {:<8.5f}".format(accuracy_score(y_age_predict, y_test)))
    # print('y_age train and predict')
    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'binary',
    #     'metric': ['binary_logloss', 'auc'],
    #     'num_leaves': 31,
    #     'min_data_in_leaf': 20,
    #     'learning_rate': 0.01,
    #     'feature_fraction': 0.8,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'nthread': -1,
    #     'verbose': -1,
    #     'seed': 1,
    #     'bagging_seed': 1,
    #     'feature_fraction_seed': 7
    # }
    # y_gender = y_gender - 1
    # x_tran, x_test, y_tran, y_test = model_selection.train_test_split(x, y_gender, test_size=0.1)
    # print(x_test.shape)
    # print('Training...')
    # trn_data = lgb.Dataset(x_tran, y_tran)
    # val_data = lgb.Dataset(x_test, y_test)
    # clf = lgb.train(params, trn_data, num_boost_round=1000, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    #
    # y_prob = clf.predict(x_test, num_iteration=clf.best_iteration)
    # max_test_score = 0
    # for i in np.arange(0.4, 0.6, 0.01):
    #     print("threshold is {}: ".format(i))
    #     y_pred = np.where(y_prob > i, 1, 0)
    #     test_score = accuracy_score(y_test, y_pred)
    #     max_test_score = max(max_test_score, test_score)
    #     print('测试集准确率：', test_score)
    #
    # print('Predicting...')
    # y_prob = clf.predict(x_predict, num_iteration=clf.best_iteration)
    # y_gender_predict = [list(x).index(max(x)) for x in y_prob]

    # 保存submission
    # submission = pd.DataFrame({'user_id': list(predict_user_id), 'predicted_age': list(y_age_predict), 'predicted_gender': list(y_gender_predict)})
    # submission.to_csv(os.path.abspath(PROJECT_PATH + '/data/submission_0518.csv'), index=False)


if __name__ == '__main__':
    allin_train_lgbm()