# -*- coding: utf-8 -*-
from tencentAdvAlgorithmContest.lgbm.w2v.w2v_adid_0517 import *
# ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry']


def save_train_test_data(tag='age', word='creative_id'):
    print('构造原始数据集')
    train_user = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/train_preliminary/user.csv'))
    train_user_click_ad_y = pkl.load(open(os.path.abspath(PROJECT_PATH + 'data/train_user_click_ad.pkl'), 'rb'))
    train_user_click_ad = train_user_click_ad_y.drop(['age', 'gender'], axis=1)
    train_user_click_ad_timesort = train_user_click_ad.sort_values(by='time')

    def add_(data):
        res = ''
        for each in data:
            res += ' ' + str(each)
        return res

    usr_id_add = train_user_click_ad_timesort.groupby('user_id').aggregate({word: add_}).reset_index()
    usr_id_add_y = pd.merge(usr_id_add, train_user, on='user_id')
    # # 去除异常用户
    # train_user_clean = pkl.load(open('data/followyu/0524/train_user_clean.pkl', 'rb'))
    # usr_id_add_y = usr_id_add_y[usr_id_add_y['user_id'].isin(tuple(train_user_clean))]
    # print('去除异常用户后，训练数据： ', len(usr_id_add_y))
    print('写 train.txt')
    with open(os.path.abspath(PROJECT_PATH + '/data/0525_8/data_0528/' + word + '_' + tag + '_train.txt'), 'w') as f_train:
        with open(os.path.abspath(PROJECT_PATH + '/data/0525_8/data_0528/' + word + '_' + tag + '_dev.txt'), 'w') as f_dev:
            for i in range(len(usr_id_add_y)):
                user_id, words, age, gender = usr_id_add_y.iloc[i]
                # gender_feature = '男' if str(gender) == '1' else '女'        # 加入性别特征
                y_col = age if tag == 'age' else gender
                if i % 10 == 0:
                    f_dev.write(words + '\t' + str(y_col) + '\n')
                else:
                    f_train.write(words + '\t' + str(y_col) + '\n')



    print('构建 test 数据集')
    test_click_log = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/test/click_log.csv'))
    test_ad = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/test/ad.csv'))
    test_user_click_ad = pd.merge(test_click_log, test_ad, on='creative_id')
    test_user_click_ad_timesort = test_user_click_ad.sort_values(by='time')

    def add_(data):
        res = ''
        for each in data:
            res += ' ' + str(each)
        return res

    usr_id_add_test = test_user_click_ad_timesort.groupby('user_id').agg({word: add_}).reset_index()
    usr_id_add_test = usr_id_add_test.sort_values('user_id')
    print('写 test.txt')
    with open(os.path.abspath(PROJECT_PATH + '/data/0525_8/data_0528/' + word + '_' + tag + '_test.txt'), 'w') as f_test:
        for i in range(len(usr_id_add_test)):
            user_id, words = usr_id_add_test.iloc[i]
            f_test.write(words + '\t' + str(0) + '\n')


if __name__ == '__main__':
    for col in ['creative_id', 'ad_id', 'product_id', 'advertiser_id', 'industry', 'product_category', 'time', 'click_times']:
        for tag in ['gender', 'age']:
            save_train_test_data(tag=tag, word=col)