# -*- coding: utf-8 -*-
import pandas as pd
import pickle as pkl
# ['time', 'user_id', 'creative_id', 'click_times', 'age', 'gender', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']


def make_statistics_feature():
    print('构造原始数据集统计特征')
    train_user_click_ad_y = pkl.load(open('data/train_user_click_ad.pkl', 'rb'))
    print('构造总点击次数')
    train_user_click_sum = train_user_click_ad_y.groupby('user_id')['click_times'].sum().reset_index()
    print('构造总点击天数')
    train_user_time_nunique = train_user_click_ad_y.groupby('user_id')['time'].nunique().reset_index()
    print('构造count')
    train_user_time_count = train_user_click_ad_y.groupby('user_id')['time'].count().reset_index()
    print('构造nuique')
    train_user_creative_id_nunique = train_user_click_ad_y.groupby('user_id')['creative_id'].nunique().reset_index()
    train_user_ad_id_nunique = train_user_click_ad_y.groupby('user_id')['ad_id'].nunique().reset_index()
    train_user_product_id_nunique = train_user_click_ad_y.groupby('user_id')['product_id'].nunique().reset_index()
    train_user_product_category_nunique = train_user_click_ad_y.groupby('user_id')['product_category'].nunique().reset_index()
    train_user_advertiser_id_nunique = train_user_click_ad_y.groupby('user_id')['advertiser_id'].nunique().reset_index()
    train_user_industry_nunique = train_user_click_ad_y.groupby('user_id')['industry'].nunique().reset_index()
    # -------------------
    feature_x_final = pd.merge(train_user_click_sum, train_user_time_nunique, on='user_id')
    feature_x_final = pd.merge(feature_x_final, train_user_time_count, on='user_id')
    feature_x_final = pd.merge(feature_x_final, train_user_creative_id_nunique, on='user_id')
    feature_x_final = pd.merge(feature_x_final, train_user_ad_id_nunique, on='user_id')
    feature_x_final = pd.merge(feature_x_final, train_user_product_id_nunique, on='user_id')
    feature_x_final = pd.merge(feature_x_final, train_user_product_category_nunique, on='user_id')
    feature_x_final = pd.merge(feature_x_final, train_user_advertiser_id_nunique, on='user_id')
    feature_x_final = pd.merge(feature_x_final, train_user_industry_nunique, on='user_id')
    feature_x_final.columns = ['user_id', 'sum_click_times', 'sum_time', 'count_time', 'creative_id_nunique', 'ad_id_nunique',
                               'product_id_nunique', 'product_category_nunique', 'advertiser_id_nunique', 'industry_nunique']
    print('构造most')
    for col in ['time', 'creative_id', 'ad_id', 'product_id', 'product_category', 'advertiser_id', 'industry']:
        user_col_click = train_user_click_ad_y.groupby(['user_id', col])['click_times'].sum().reset_index()
        user_col_click.sort_values('click_times', ascending=False, inplace=True)
        user_col_click.drop_duplicates('user_id', keep='first', inplace=True)
        user_col_click.columns = ['user_id', 'most_' + col, 'most_' + col + '_click_times']
        feature_x_final = pd.merge(feature_x_final, user_col_click, on='user_id')

    feature_x_final.sort_values('user_id', inplace=True)
    pkl.dump(feature_x_final, open('/data/0519/feature_x_final.pkl', 'wb'))
    print('feature_x_final 完成')


if __name__ == '__main__':
    make_statistics_feature()