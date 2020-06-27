# -*- coding: utf-8 -*-
import pickle as pkl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
import lightgbm as lgb
from gensim.models import word2vec
import numpy as np


def tfidf_feature_eng():
    print('读取训练/测试集原始数据集')
    train_user_click_ad_y = pkl.load(open('data/train_user_click_ad.pkl', 'rb'))
    train_user_click_ad = train_user_click_ad_y.drop(['age', 'gender'], axis=1)

    test_click_log = pd.read_csv('data/test/click_log.csv')
    test_ad = pd.read_csv('data/test/ad.csv')
    test_user_click_ad = pd.merge(test_click_log, test_ad, on='creative_id')
    train_test_click_ad = pd.concat([train_user_click_ad, test_user_click_ad])

    train_test_user_creative_text = train_test_click_ad.groupby('user_id').agg({'creative_id': lambda x: ' '.join([str(_) for _ in x]).strip()}).reset_index()
    train_test_click_ad_user_id = train_test_user_creative_text['user_id']
    print('tfidf')
    tv = TfidfVectorizer(min_df=8000)  # 创建词袋数据结构
    tv_fit = tv.fit_transform(list(train_test_user_creative_text['creative_id'].values))
    tv_fit_array = tv_fit.toarray()
    tv_fit_array_pd = pd.DataFrame(tv_fit_array)
    train_test_click_ad_user_id = train_test_click_ad_user_id.reset_index(drop=True)
    tv_fit_array_userid = pd.concat([train_test_click_ad_user_id, tv_fit_array_pd], axis=1)
    print(tv_fit_array_userid.shape)
    print(tv_fit_array_userid.columns)

    train_user_static_feature_tfidf = tv_fit_array_userid[tv_fit_array_userid['user_id'].isin(tuple(train_user_click_ad['user_id'].values))]
    test_user_static_feature_tfidf = tv_fit_array_userid[tv_fit_array_userid['user_id'].isin(tuple(test_user_click_ad['user_id'].values))]

    train_user_static_feature = pkl.load(open('data/followyu/0524/train_user_static_feature.pkl', 'rb'))
    test_user_static_feature = pkl.load(open('data/followyu/0524/test_user_static_feature.pkl', 'rb'))

    train_user_static_feature = pd.merge(train_user_static_feature_tfidf, train_user_static_feature, on='user_id')
    test_user_static_feature = pd.merge(test_user_static_feature_tfidf, test_user_static_feature, on='user_id')
    print(train_user_static_feature.shape)
    print(test_user_static_feature.shape)
    usr_word_id_emb_expand_y = make_data_train('ad_id', 'ad_id_w2vmodel_0520', 128)
    usr_word_id_emb_expand = usr_word_id_emb_expand_y.drop(['age', 'gender'], axis=1)
    train_user_static_feature = pd.merge(train_user_static_feature, usr_word_id_emb_expand, on='user_id')
    print(train_user_static_feature.shape)
    print(test_user_static_feature.shape)
    # pkl.dump(train_user_static_feature, open('data/followyu/0524/train_user_static_feature_tfidf50000_stastic.pkl', 'wb'))   大于4g不能存取
    # pkl.dump(test_user_static_feature, open('data/followyu/0524/test_user_static_feature_tfidf50000_stastic.pkl', 'wb'))
    print('tfidf_feature_eng 完成')
    return train_user_static_feature, test_user_static_feature


def make_data_train(word, word2vec_model_name='creativeid300_w2vmodel_0518', w2v_size=300):
    print('开始load 训练集')
    train_user = pd.read_csv('data/train_preliminary/user.csv')
    model = word2vec.Word2Vec.load('data/0520/' + word2vec_model_name)
    train_user_click_ad_y = pkl.load(open('data/train_user_click_ad.pkl', 'rb'))

    print('开始emb')
    usr_creative_id_list = train_user_click_ad_y.groupby('user_id').aggregate({word: lambda x: [str(_) for _ in x]}).reset_index()
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


def run_lgbm(train_user_static_feature=None, test_user_static_feature=None):
    if not isinstance(train_user_static_feature, pd.DataFrame) and not isinstance(train_user_static_feature, pd.DataFrame):
        train_user_static_feature = pkl.load(open('data/followyu/0524/train_user_static_feature_tfidf50000_stastic.pkl', 'rb'))
        test_user_static_feature = pkl.load(open('data/followyu/0524/test_user_static_feature_tfidf50000_stastic.pkl', 'rb'))
    x_train = train_user_static_feature.drop('user_id', axis=1)
    train_user = pd.read_csv('data/train_preliminary/user.csv')
    train_user_static_feature_y = pd.merge(train_user_static_feature, train_user, on='user_id')
    y_gender = train_user_static_feature_y['gender']
    y_age = train_user_static_feature_y['age']

    x_predict = test_user_static_feature.drop('user_id', axis=1)
    user_id_predict = test_user_static_feature['user_id']

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

    print('开始训练性别')
    x_tran, x_test, y_tran, y_test = model_selection.train_test_split(x_train, y_gender, test_size=0.2)
    print(x_test.shape)
    print('Training...')
    trn_data = lgb.Dataset(x_tran, y_tran)
    val_data = lgb.Dataset(x_test, y_test)
    clf = lgb.train(params, trn_data, num_boost_round=1000, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    print('Predicting...')
    y_prob = clf.predict(x_predict, num_iteration=clf.best_iteration)
    y_gender_predict = [list(x).index(max(x)) for x in y_prob]

    print('开始训练年龄')
    x_tran, x_test, y_tran, y_test = model_selection.train_test_split(x_train, y_age, test_size=0.2)
    print(x_test.shape)
    print('Training...')
    trn_data = lgb.Dataset(x_tran, y_tran)
    val_data = lgb.Dataset(x_test, y_test)
    clf = lgb.train(params, trn_data, num_boost_round=1000, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
    print('Predicting...')
    y_prob = clf.predict(x_predict, num_iteration=clf.best_iteration)
    y_age_predict = [list(x).index(max(x)) for x in y_prob]

    # 保存submission
    submission = pd.DataFrame({'user_id': list(user_id_predict), 'predicted_age': list(y_age_predict), 'predicted_gender': list(y_gender_predict)})
    submission.to_csv('data/followyu/submission_0524.csv', index=False)


if __name__ == '__main__':
    train_user_static_feature, test_user_static_feature = tfidf_feature_eng()
    run_lgbm(train_user_static_feature, test_user_static_feature)