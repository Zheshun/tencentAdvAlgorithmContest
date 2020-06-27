# -*- coding: utf-8 -*-
import os
import pandas as pd
import pickle as pkl

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
PROJECT_PATH = os.path.abspath(CURRENT_PATH + '/../')


train_user = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/train_preliminary/user.csv'))
train_click_log = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/train_preliminary/click_log.csv'))
train_ad = pd.read_csv(os.path.abspath(PROJECT_PATH + '/data/train_preliminary/ad.csv'))

train_user_click = pd.merge(train_click_log, train_user, on='user_id')
train_user_click_ad = pd.merge(train_user_click, train_ad, on='creative_id')
print('train_user_click_ad 完成')
train_user_click_ad_y = pkl.dump(train_user_click_ad, open('data/train_user_click_ad.pkl', 'wb'))
