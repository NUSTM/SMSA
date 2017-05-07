#!/usr/bin/env cpp
# -*- coding: utf-8 -*-
# @Date    : 2017-01-25 10:26:59
# @Author  : JieJ (jiej1992@163.com)

import os,sys
import params
import random
import pytc
import tools
from performance import performance
from Mbsa_new import Mbsa

if __name__ == '__main__':
    term_weight = 'BOOL'
    train_params = {
        'token' : ['neg_fenci', 'pos_fenci'],
        'pos' : [],
        'tag' : [],
        'class_name' : ['neg', 'pos'],
        'raw_dir': '',
        'samp_dir': '',
        'term_set_dir': '',
        'model_file_dir': '',
    }

    test_params = {
        'token' : ['test_fenci'],
        'pos' : ['test_pos'],
        'tag' : ['test_tag'],
        'class_name' : ['test'],
        'raw_dir': '',
        'samp_dir': '',
        'term_set_dir': '',
        'model_file_dir': '',
        'result_file_dir': ''
    }

    classifier_lst = ['lg']
    class_dict = {'1':'neg','2':'pos'}

    data = sys.argv[1]
    ngram = sys.argv[2]
    rule_feature = int(sys.argv[3])
    if ngram == 'bis':
        ngram_dict = params.bigram_dict
    else:
        ngram_dict = params.unigram_dict

    input_dir = 'Data' + os.sep + data + '_THULAC'
    output_dir = input_dir + '_nfolds'
    fold_num = 5
    classifier_list = ['svm']
    class_dict = {'1':'neg', '2':'pos'}
    test = Mbsa(ngram_dict, term_weight, rule_feature)
    test.N_folds_samps(input_dir, fold_num, train_params, test_params)
    test.N_folds_validation(input_dir, fold_num, classifier_list, train_params, test_params)
    for c in classifier_list:
        result_dict = performance.demo_cv_performance(output_dir, fold_num, class_dict, c)
        ss = ''
        for key in ['macro_r','macro_p','macro_f1','acc']:
            ss += str(round(result_dict[key]*100,2))+'%\t'
        print ss.rstrip('\t')
