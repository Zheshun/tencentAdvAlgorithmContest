# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
from tencentAdvAlgorithmContest.lgbm.w2v.w2v_adid_0517 import *
from tencentAdvAlgorithmContest.nn.train_eval import train, init_network
from tencentAdvAlgorithmContest.nn.utils import build_dataset, build_iterator, get_time_dif


args_model = 'TextRNN' #  TextRNN, TextRNN_Att
args_word = True  # 'True for word, False for char'

if __name__ == '__main__':
    tag = 'age'
    dataset = os.path.abspath(PROJECT_PATH + '/data/0525_8/')  # 数据集
    model_name = args_model
    x = import_module('models.' + model_name)
    config = x.Config(dataset, tag)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(dataset, config, args_word, tag)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

    # 整合submission
    # submission_demo_pd = pd.read_csv('/Users/zzs/PycharmProjects/month5/tencentAdvAlg/data/0525_8/submission_1.367.csv')
    # age_pd = pd.read_csv('/Users/zzs/PycharmProjects/month5/tencentAdvAlg/data/0525_8/age_47.288.csv')
    # gender_pd = pd.read_csv('/Users/zzs/PycharmProjects/month5/tencentAdvAlg/data/0525_8/gender_94.247.csv')
    # submission = pd.DataFrame({'user_id': list(submission_demo_pd['user_id']), 'predicted_age': list(age_pd['predicted_age']), 'predicted_gender': list(gender_pd['predicted_gender'])})
    # submission.to_csv('/Users/zzs/PycharmProjects/month5/tencentAdvAlg/data/0525_8/submission.csv', index=False)
