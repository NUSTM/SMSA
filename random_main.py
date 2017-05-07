# -*- coding: utf-8 -*-
# from __future__ import division
import os, time, sys, math
import random
import pytc
import pick_samps
import params
import numpy as np

from Mbsa_new import Mbsa
from performance import performance

if __name__ == '__main__':
    test_dir_dict = {
        'COAE2014': 'Data/COAE2014_THULAC'
    }

    test = Mbsa(params.ngram_dict, params.term_weight, params.rule_feature)

    classifier_lst = ['lg']
    items = ['macro_f1','acc']
    class_dict = {'1':'neg','2':'pos'}

    input_dir = 'corpus/zh_distant_data'

    topic_id = 'train-distant-zh'
    output_dir = 'bagging' + os.sep + topic_id

    src_neg_fname = 'neg_fenci'
    src_pos_fname = 'pos_fenci'

    neg_lst = open(input_dir+os.sep+src_neg_fname).readlines()
    pos_lst = open(input_dir+os.sep+src_pos_fname).readlines()
    neg_index_lst = range(len(neg_lst))
    pos_index_lst = range(len(pos_lst))
    random.shuffle(neg_index_lst)
    random.shuffle(pos_index_lst)

    tar_neg_fname = 'neg_fenci'
    tar_pos_fname = 'pos_fenci'
    test_data_lst = ['COAE2014']
    # test_data_lst = ['twitter2013','twitter2014','twitter2015','twitter2016',
    # 'elec', 'kitchen', 'book', 'dvd']

    tic = time.time()
    samp_num_list = [1000]

    for round_num, j in enumerate(samp_num_list):
        # 用来存储每轮的性能，便于后面求平均性能
        performance_dict = {}
        for data_name in test_dir_dict:
            performance_dict[data_name] = {}.fromkeys(items,0)

        # 存储每轮中的结果标签， 结果概率
        label_result_dict, prob_result_dict = {}, {}
        for x in test_dir_dict.keys():
            label_result_dict[x] = []
            prob_result_dict[x] = []

        samp_num = j
        for k in range(1, 6):
            print 'generating random samps for',round_num+1,'th round......'
            print 'generating random samps for',k,'th time......'

            if os.path.exists(input_dir) and os.path.exists(output_dir):
                pick_samps.random_sample_lst(neg_lst, output_dir, tar_neg_fname, samp_num)
                pick_samps.random_sample_lst(pos_lst, output_dir, tar_pos_fname, samp_num)

                # pick_samps.random_sample_lst_2(neg_lst, neg_index_lst, output_dir, tar_neg_fname, samp_num, start_index=(k-1)*samp_num)
                # pick_samps.random_sample_lst_2(pos_lst, pos_index_lst, output_dir, tar_pos_fname, samp_num, start_index=(k-1)*samp_num)

            else:
                print "No such file or directory"

            # '''train & test'''
            train_dir = 'bagging' + os.sep+ topic_id
            train_params = params.train_params
            train_params['raw_dir'] = train_dir
            train_params['samp_dir'] = train_dir
            train_params['term_set_dir'] = train_dir
            train_params['model_file_dir'] = train_dir
            test.gen_train_samps(train_params)

            for c in classifier_lst:
                print c, "learning..."
                model_fname = c + '.model'
                params.classifier_learn[c](train_params['samp_dir'], train_params['model_file_dir'], model_fname = model_fname)

            test_params = params.test_params

            for data_name in test_data_lst:
                print data_name
                test_dir = test_dir_dict[data_name]
                test_output_dir = test_dir + os.sep + 'ml_ensemble'
                if not os.path.exists(test_output_dir):
                    os.mkdir(test_output_dir)

                test_params['raw_dir'] = test_dir
                test_params['samp_dir'] = test_output_dir
                test_params['term_set_dir'] = train_dir
                test_params['model_file_dir'] = train_dir
                test_params['result_file_dir'] = test_output_dir
                test.gen_test_samps(test_params)

                label = [x.strip() for x in open(test_dir + os.sep + 'test_label').readlines()]

                # 注意：mallet中的模型在预测svmlight数据格式时，不能够见到训练集以外的标签
                for index, c in enumerate(classifier_lst):
                    print c, "predicting..."
                    model_fname = c + '.model'
                    result_fname = c + '.result'
                    params.classifier_predict[c](test_params['samp_dir'], test_params['model_file_dir'],
                        test_params['result_file_dir'], model_fname = model_fname, result_fname = result_fname)

                    start = 1
                    if c == 'nb':
                        start = 0

                    # 将本轮结果标签存储
                    result = [x.strip() for x in open(test_params['result_file_dir'] + \
                        os.sep + result_fname).readlines()[start:]]
                    preds = [x.split()[0] for x in result]
                    label_result_dict[data_name].append(preds)

                    # 将本轮结果概率存储
                    mat = np.array( [[float(x) for x in line.strip().split()[1:]] for line in result])
                    prob_result_dict[data_name].append(mat)

                    # 计算这一次的分类性能，并将分类性能加入到该测试集的性能字典中去
                    result_dict = performance.demo_performance(preds,label,class_dict)
                    for key in items:
                        performance_dict[data_name][key] += result_dict[key]

            predict_num = k * len(classifier_lst)

            if predict_num >= 1:
                vote_threshold = math.ceil(predict_num/2.0)
                print "投票胜出阈值：", vote_threshold
                F_lst = []
                avg_F_lst = []
                ACC_lst = []
                for data_name in test_data_lst:
                    tmp_lst = []     # 用于存储单个、投票、加权3种性能
                    print data_name,"predict_num=",predict_num
                    path = test_dir_dict[data_name]
                    label = [x.strip() for x in open(path + os.sep + 'test_label').readlines()]

                    '''1. 平均性能计算'''
                    ss = ''
                    for key in items:
                        ss += str(round(performance_dict[data_name][key]*100/predict_num, 2))+'%\t'
                    ss = ss.rstrip('\t')
                    tmp_lst.append(ss)
                    # avg_F =  ss.split('\t')[-2]
                    # avg_F_lst.append(avg_F)
                    print "平均性能：\n" + ss

                    '''2. 通过投票方式选出最终结果'''
                    vote_result = []
                    all_result = map(list, zip(*label_result_dict[data_name]))    # 实现二维list的转置
                    # 这里是2类别分类，暂时采用简单的方法投票
                    for item in all_result:
                        if item.count('1') >= vote_threshold:
                            vote_result.append('1')
                        else:
                            vote_result.append('2')

                    # 投票结果性能计算、存储
                    result_dict = performance.demo_performance(vote_result, label, class_dict)
                    ss = ''
                    for key in items:
                        ss += str(round(result_dict[key]*100, 2))+'%\t'
                    ss = ss.rstrip('\t')
                    tmp_lst.append(ss)
                    print "投票性能：\n" + ss

                    '''3. 通过概率融合方式选出最终结果'''
                    mix_prob_mat = np.array(np.zeros(prob_result_dict[data_name][0].shape))
                    for prob_mat in prob_result_dict[data_name]:
                        mix_prob_mat += prob_mat
                    mix_result = list(np.argmax(mix_prob_mat,1))
                    label_dict = {0: '1', 1: '2'}
                    mix_result = [label_dict[x] for x in mix_result]

                    # 融合结果存储
                    if not os.path.exists(path + os.sep + 'ml_prob_mix'):
                        os.mkdir(path + os.sep + 'ml_prob_mix')
                    with open(path + os.sep + 'ml_prob_mix' + os.sep + 'mix_lg_result.txt', 'w') as xs:
                        xs.write('labels 1 2' + '\n')
                        for i in range(len(mix_result)):
                            xs.write(mix_result[i] + ' ' + str(mix_prob_mat[i][0]/k) + ' ' + str(mix_prob_mat[i][1]/k) + '\n')

                    with open(path + os.sep + 'ml_prob_mix' + os.sep + 'mix_result.txt', 'w') as xs:
                        xs.writelines([x + '\n' for x in mix_result])

                    # 概率融合性能计算
                    result_dict = performance.demo_performance(mix_result, label, class_dict)
                    ss = ''
                    for key in items:
                        ss += str(round(result_dict[key]*100, 2))+'%\t'
                    print data_name
                    ss = ss.rstrip('\t')
                    tmp_lst.append(ss)

                    print "概率融合性能："
                    print ss
                    # P = round((result_dict['p_neg']+result_dict['p_pos'])*50,2)
                    # R = round((result_dict['r_neg']+result_dict['r_pos'])*50,2)
                    F =  round(result_dict['macro_f1']*100,2)
                    ACC = round(result_dict['acc']*100,2)

                    F_lst.append(str(F))
                    ACC_lst.append(str(ACC))
    print time.time() - tic, "secs."
