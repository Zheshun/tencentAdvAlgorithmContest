# -*- coding: utf-8 -*-
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle as pkl
from sklearn import model_selection
import lightgbm as lgb


def aggregate_features(df, prefix):
    agg_func = {
        'time': ['count', 'nunique'],
        'creative_id': ['nunique'],
        'click_times': ['nunique', 'mean', 'max', 'min', 'std'],
        'ad_id': ['nunique'],
        'product_id': ['nunique'],
        'product_category': ['nunique'],
        'advertiser_id': ['nunique'],
        'industry': ['nunique']
    }

    agg_df = df.groupby(['user_id']).agg(agg_func)
    agg_df.columns = [prefix + '_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(drop=False, inplace=True)
    return agg_df


def base_feature_eng():
    print('读取训练/测试集原始数据集')
    train_user_click_ad_y = pkl.load(open('data/train_user_click_ad.pkl', 'rb'))

    test_click_log = pd.read_csv('data/test/click_log.csv')
    test_ad = pd.read_csv('data/test/ad.csv')
    test_user_click_ad = pd.merge(test_click_log, test_ad, on='creative_id')
    print('数据提取特征')
    train_user_static_feature = aggregate_features(train_user_click_ad_y, '0524')
    train_user_static_feature.sort_values('user_id', inplace=True)

    test_user_static_feature = aggregate_features(test_user_click_ad, '0524')
    test_user_static_feature.sort_values('user_id', inplace=True)
    print('基础统计特征完成')
    print('训练数据异常去除')
    train_user_static_feature = train_user_static_feature[train_user_static_feature['0524creative_id_nunique'] < 150]
    test_user_static_feature_clean = test_user_static_feature[test_user_static_feature['0524creative_id_nunique'] < 150]
    pkl.dump(train_user_static_feature['user_id'].values.tolist(), open('data/followyu/0524/train_user_clean.pkl', 'wb'))
    pkl.dump(test_user_static_feature_clean['user_id'].values.tolist(),open('data/followyu/0524/test_user_clean.pkl', 'wb'))

    pkl.dump(train_user_static_feature, open('data/followyu/0524/train_user_static_feature.pkl', 'wb'))
    pkl.dump(test_user_static_feature, open('data/followyu/0524/test_user_static_feature.pkl', 'wb'))
    return train_user_static_feature, test_user_static_feature


def run_lgbm():
    train_user_static_feature = pkl.load(open('data/followyu/0524/train_user_static_feature.pkl', 'rb'))
    test_user_static_feature = pkl.load(open('data/followyu/0524/test_user_static_feature.pkl', 'rb'))
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
    train_user_static_feature, test_user_static_feature = base_feature_eng()
    run_lgbm()