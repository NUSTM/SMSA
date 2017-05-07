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
    max_val = 1000
    min_val = -1000
    for key in input_dict:
        if input_dict[key] >= max_val:
            input_dict[key] = 1.0
        elif input_dict[key] <= min_val:
            input_dict[key] = -1.0
        else:
            new_val = -1 + (2 * (input_dict[key] - min_val)/(max_val - min_val))
            input_dict[key] = new_val

# def feature_selection_chi(df_class, df_term_class, fs_num=0, fs_class=-1):
#     #df_class:不同类别的文档数
#     #df_term_class: 每个词在各个类别下的文档数
#     #{term1:[class1:num,class2:num,..],..}
#     term_set = df_term_class.keys()
#     term_score_dict = {}.fromkeys(term_set)
#     for term in term_set:
#         df_list = df_term_class[term]
#         class_set_size = len(df_list)
#         cap_n = sum(df_class)
#         score_list = []
#         for class_id in range(class_set_size):
#             cap_a = df_list[class_id]
#             cap_b = sum(df_list) - cap_a
#             cap_c = df_class[class_id] - cap_a
#             cap_d = cap_n - cap_a - cap_c - cap_b
#             cap_nu = float(cap_a * cap_d - cap_c * cap_b)
#             cap_x1 = cap_nu / ((cap_a + cap_c) * (cap_b + cap_d))
#             cap_x2 = cap_nu / ((cap_a+cap_b) * (cap_c+cap_d))
#             score = cap_n * cap_x1 * cap_x2
#             score_list.append(score)
#         if fs_class == -1:
#             term_score = max(score_list)
#         else:
#             term_score = score_list[fs_class]
#         term_score_dict[term] = term_score
#     term_score_list = term_score_dict.items()
#     term_score_list.sort(key=lambda x: -x[1])
#     term_set_rank = [x[0] for x in term_score_list]
#     if fs_num == 0:
#         term_set_fs = term_set_rank
#     else:
#         term_set_fs = term_set_rank[:fs_num]
#     return term_set_fs, term_score_dict

def cal_chi(cap_a, cap_b, cap_c, cap_d, cap_n):
    cap_nu = cap_a * cap_d - cap_c * cap_b
    cap_x1 = cap_nu / ((cap_a + cap_c) * (cap_b + cap_d))
    cap_x2 = cap_nu / ((cap_a + cap_b) * (cap_c + cap_d))
    score = cap_n * cap_x1 * cap_x2
    return score

if __name__ == '__main__':
    ngram_list = ['uni']
    metric = 'DF'

    input_dir = 'corpus' + os.sep + 'unlabeled_data'
    pos_fenci_lines = [x.strip() for x in open(input_dir + os.sep + 'pos_fenci').readlines()]
    neg_fenci_lines = [x.strip() for x in open(input_dir + os.sep + 'neg_fenci').readlines()]

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
        cap_n = pos_df + neg_df
        for index, term in enumerate(term_set):
            cap_a = term_pos_freq[term]
            cap_b = term_neg_freq[term]
            cap_c = pos_df - cap_a
            # pos_cap_d = pos_df + neg_df - pos_cap_a - pos_cap_b - pos_cap_c
            cap_d = neg_df - cap_b

            if(cap_a == 0 and cap_b == 0):
                term_senti_dict[term] = 0
            else:
                chi_val = cal_chi(cap_a, cap_b, cap_c, cap_d, cap_n)
                tmp = cap_a * neg_df - cap_b * pos_df
                alpha = abs(tmp) / (cap_a * neg_df + cap_b * pos_df)
                score = chi_val * alpha
                if tmp > 0:
                    term_senti_dict[term] = score
                elif tmp < 0:
                    term_senti_dict[term] = -score
                else:
                    print "funny"
                    pass

        normalize_dict(term_senti_dict)
        term_senti_tuple = sorted(term_senti_dict.iteritems(), key=lambda x:abs(x[1]), reverse=True)
        print term_senti_tuple[0][0], term_senti_tuple[0][1]
        print term_senti_tuple[1][0], term_senti_tuple[1][1]

        output_dir = 'DICT' + os.sep + 'distant_chi_' + metric

        with open(output_dir + os.sep + 'senti_' + ngram + '_all.txt', 'w') as xs:
            for item in term_senti_tuple:
                if item[1] != 0:
                    xs.write(item[0] + '\t' + str(item[1]) + '\n')

