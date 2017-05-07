# -*- coding: utf-8 -*-
"""
根据弱标注语料生成远程监督词典
可控参数：
1. N-GRAM
2. TF or DF
"""

from __future__ import division
import os
import math
import re

import pytc

def get_term_set(fenci_lines):
    term_set = set()
    for line in fenci_lines:
        line_set = set(line.strip().split())
        term_set.update(line_set)

    return list(term_set)

def gen_term_set(doc_terms_list):
    term_set = set()
    for lst in doc_terms_list:
        term_set.update(lst)
    return list(term_set)

def normalize_dict(input_dict):
    # val_lst = input_dict.itervalues()
    max_val = 40
    min_val = -40
    for key in input_dict:
        if input_dict[key] >= max_val:
            input_dict[key] = 1.0
        elif input_dict[key] <= min_val:
            input_dict[key] = -1.0
        else:
            new_val = -1 + (2 * (input_dict[key] - min_val)/(max_val - min_val))
            input_dict[key] = new_val

def cal_wllr(p_t_c, p_t_not_c):

    # score = math.log(p_t_c / p_t_not_c)
    score = p_t_c * math.log(p_t_c / p_t_not_c)
    return score

def feature_selection_wllr(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
#       doc_set_size = len(df_class)
        cap_n = sum(df_class)
        term_set_size = len(df_term_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            p_t_c = (cap_a + 1E-6) / (cap_a + cap_c + 1E-6*term_set_size)
            p_t_not_c = (cap_b + 1E-6)/(cap_b + cap_d + 1E-6*term_set_size)
            score = p_t_c * math.log(p_t_c / p_t_not_c)
            score_list.append(score)
        if fs_class == -1:
            term_score = max(score_list)
        else:
            term_score = score_list[fs_class]
        term_score_dict[term] = term_score
    term_score_list = term_score_dict.items()
    term_score_list.sort(key=lambda x: -x[1])
    term_set_rank = [x[0] for x in term_score_list]
    if fs_num == 0:
        term_set_fs = term_set_rank
    else:
        term_set_fs = term_set_rank[:fs_num]
    return term_set_fs, term_score_dict

if __name__ == '__main__':
    ngram_list = ['uni']
    metric = 'DF'

    input_dir = 'corpus' + os.sep + 'unlabeled_data'
    pos_fenci_lines = [re.sub('#.*?#', '', x.strip()) for x in open(input_dir + os.sep + 'pos_fenci').readlines()]
    neg_fenci_lines = [re.sub('#.*?#', '', x.strip()) for x in open(input_dir + os.sep + 'neg_fenci').readlines()]

    for ngram in ngram_list:
        pos_terms_list = pytc.gen_N_gram(pos_fenci_lines, ngram)
        neg_terms_list = pytc.gen_N_gram(neg_fenci_lines, ngram)
        term_set = gen_term_set(pos_terms_list + neg_terms_list)

        term_pos_freq = {}.fromkeys(term_set, 0)    # 词在情感正向类中的词频或者文档频
        term_neg_freq = {}.fromkeys(term_set, 0)    # 词在情感负向类中的词频或者文档频

        pos_df = len(pos_terms_list)
        neg_df = len(neg_terms_list)

        print "pos_df=", pos_df
        print "neg_df=", neg_df

        for lst in pos_terms_list:
            if metric == 'DF':
                lst = list(set(lst))
            for term in lst:
                term_pos_freq[term] += 1

        for lst in neg_terms_list:
            if metric == 'DF':
                lst = list(set(lst))
            for term in lst:
                term_neg_freq[term] += 1

        term_senti_dict = {}
        term_set_size = len(term_set)
        df_sum = pos_df + neg_df

        for index, term in enumerate(term_set):
            p_t_pos = (term_pos_freq[term] + 1E-6) / (pos_df + 1E-6 * term_set_size)
            p_t_neg = (term_neg_freq[term] + 1E-6) / (neg_df + 1E-6 * term_set_size)
            score = cal_wllr(p_t_pos, p_t_neg) - cal_wllr(p_t_neg, p_t_pos)
            term_senti_dict[term] = score

        # normalize_dict(term_senti_dict)
        term_senti_tuple = sorted(term_senti_dict.iteritems(), key=lambda x:abs(x[1]), reverse=True)
        print term_senti_tuple[0][0], term_senti_tuple[0][1]
        print term_senti_tuple[1][0], term_senti_tuple[1][1]

        output_dir = 'DICT' + os.sep + 'distant_wllr_' + metric
        # output_dir = 'DICT' + os.sep + 'distant_llr_' + metric
        with open(output_dir + os.sep + 'senti_' + ngram + '_all.txt', 'w') as xs:
            for item in term_senti_tuple:
                if item[1] != 0:
                    xs.write(item[0] + '\t' + str(item[1]) + '\n')
