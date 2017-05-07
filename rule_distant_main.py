# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 21:19:43 2015

@author: JieJ
"""

import os
import sys

import tools
import pytc
import numpy as np

from performance import performance

def distant_dict_score(doc, senti_distant_dict):
    if len(doc) == 0:
        return 0
    score = 0
    doc = list(set(doc))
    for term in doc:
        if term in senti_distant_dict:
            score += senti_distant_dict[term]
    return score

def distant_dict_score_dicrete(doc, senti_distant_dict):
    if len(doc) == 0:
        return 0
    score = 0
    # doc = list(set(doc))
    for term in doc:
        if term in senti_distant_dict:
            if senti_distant_dict[term] > 0:
                score += 1
            else:
                score -= 1
    return score

if __name__ == '__main__':
    dict_fname = 'DICT' + os.sep + 'ZH_PMI.txt'
    ngram_list = ['uni']
    print "loading lexicon......"
    senti_dict = tools.load_lexicon(dict_fname, float)
    print "lexicon loaded!"

    base_dir = 'Data'
    test_data_list = ['jd_thulac']

    F_lst = []
    for test_data_name in test_data_list:
        test_dir = base_dir + os.sep + test_data_name
        output_dir = test_dir + os.sep + 'rule_distant'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        fenci_fname = 'test_fenci'
        rule_score_fname = 'rule_score.txt'
        rule_result_fname = 'rule_result.txt'

        test_fenci_lines = [x.strip() for x in open(test_dir + os.sep + fenci_fname).readlines()]
        print "一共" + str(len(test_fenci_lines)) + "篇文档......"

        final_score = [0.0] * len(test_fenci_lines)

        terms_list = pytc.gen_N_gram(test_fenci_lines, 'uni')
        for j in range(len(terms_list)):
            score = distant_dict_score(terms_list[j], senti_dict)
            final_score[j] += score

        tools.write_score_file(final_score, output_dir + os.sep + rule_score_fname)

        num = 0
        tools.classify_2_way(output_dir + os.sep + rule_score_fname, output_dir + os.sep + rule_result_fname, num)

        '''performance'''
        result = [x.strip() for x in open(output_dir + os.sep + 'rule_result.txt').readlines()]
        label = [x.strip() for x in open(test_dir + os.sep + 'test_label').readlines()]
        class_dict = {'1':'neg','2':'pos'}
        result_dict = performance.demo_performance(result,label,class_dict)

        ss = ''
        for key in ['macro_f1']:
            ss += str(round(result_dict[key]*100,2))+'%\t'
        ss = ss.rstrip('\t')
        print ss
        F_lst.append(ss)
    with open('rule_result.txt', 'a') as xs:
        xs.write('\t'.join(F_lst)+'\n')
    print 'over'
