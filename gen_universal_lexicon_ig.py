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
import random

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

def cal_ig(cap_a, cap_b, cap_c, cap_d, cap_n):
    p_c = float(cap_a + cap_c) / cap_n
    p_t = float(cap_a + cap_b) / cap_n
    p_nt = 1 - p_t
    p_c_t = (cap_a + 1.0) / (cap_a + cap_b + 2)
    p_c_nt = (cap_c + 1.0) / (cap_c + cap_d + 2)
    # t_entropy = -p_t * p_c_t * math.log(p_c_t)
    # not_t_entropy = -p_nt * p_c_nt * math.log(p_c_nt)
    t_entropy = -p_c_t * math.log(p_c_t)
    not_t_entropy = -p_c_nt * math.log(p_c_nt)
    ig_val = - p_c * math.log(p_c) - p_t * t_entropy - p_nt * not_t_entropy
    return t_entropy, not_t_entropy, ig_val

# def feature_selection_ig(df_class, df_term_class, fs_num=0, fs_class=-1):
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
#             p_c = float(cap_a + cap_c) / cap_n
#             p_t = float(cap_a + cap_b) / cap_n
#             p_nt = 1 - p_t
#             p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
#             p_c_nt = (cap_c + 1.0) / (cap_c + cap_d + class_set_size)
#             score = - p_c * math.log(p_c) + p_t * p_c_t * math.log(p_c_t) + \
#                 p_nt * p_c_nt * math.log(p_c_nt)
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


if __name__ == '__main__':
    ngram_list = ['uni']
    metric = 'DF'
    # metric = 'TF'

    input_dir = 'corpus' + os.sep + 'unlabeled_data'
    # input_dir = 'corpus' + os.sep + 'book2'
    pos_fenci_lines = [re.sub('#.*?#', '', x.strip()) for x in open(input_dir + os.sep + 'pos_fenci').readlines()]
    neg_fenci_lines = [re.sub('#.*?#', '', x.strip()) for x in open(input_dir + os.sep + 'neg_fenci').readlines()]
    random.shuffle(pos_fenci_lines)
    pos_fenci_lines = pos_fenci_lines[:len(neg_fenci_lines)]

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

        pos_senti_dict = {}
        neg_senti_dict = {}

        cap_n = pos_df + neg_df
        for index, term in enumerate(term_set):
            # if index > 20:
            #     break
            sum_term_df = term_pos_freq[term] + term_neg_freq[term]
            # 先求该词对正向类别带来的信息增益
            cap_a = term_pos_freq[term]
            cap_b = sum_term_df - cap_a
            cap_c = pos_df - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            t_pos_entropy, not_t_pos_entropy, ig_pos = cal_ig(cap_a, cap_b, cap_c, cap_d, cap_n)

            # 再求该词对负向类别带来的信息增益
            cap_a = term_neg_freq[term]
            cap_b = sum_term_df - cap_a
            cap_c = neg_df - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            t_neg_entropy, not_t_neg_entropy, ig_neg = cal_ig(cap_a, cap_b, cap_c, cap_d, cap_n)

            pos_degree = not_t_pos_entropy - t_pos_entropy
            neg_degree = not_t_neg_entropy - t_neg_entropy
            pos_senti_dict[term] = pos_degree
            neg_senti_dict[term] = neg_degree


            pos_degree = t_pos_entropy + not_t_neg_entropy
            neg_degree = not_t_pos_entropy + t_neg_entropy
            ig = ig_pos + ig_neg

            if ig > 0:
                ratio = abs(neg_degree - pos_degree) / (neg_degree + pos_degree)
                if pos_degree < neg_degree:
                    term_senti_dict[term] = ratio * ig
                else:
                    term_senti_dict[term] = - ratio * ig


        # normalize_dict(term_senti_dict)
        # pos_senti_tuple = sorted(pos_senti_dict.iteritems(), key=lambda x:x[1], reverse=True)
        # neg_senti_tuple = sorted(neg_senti_dict.iteritems(), key=lambda x:x[1], reverse=True)

        term_senti_tuple = sorted(term_senti_dict.iteritems(), key=lambda x:abs(x[1]), reverse=True)
        print term_senti_tuple[0][0], term_senti_tuple[0][1]
        print term_senti_tuple[1][0], term_senti_tuple[1][1]

        output_dir = 'DICT' + os.sep + 'distant_ig_' + metric
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # with open(output_dir + os.sep + 'pos_words.txt', 'w') as xs:
        #     for item in pos_senti_tuple:
        #         if item[1] != 0:
        #             xs.write(item[0] + '\t' + str(item[1]) + '\n')

        # with open(output_dir + os.sep + 'neg_words.txt', 'w') as xs:
        #     for item in neg_senti_tuple:
        #         if item[1] != 0:
        #             xs.write(item[0] + '\t' + str(item[1]) + '\n')

        with open(output_dir + os.sep + 'senti_' + ngram + '_all.txt', 'w') as xs, \
        open(output_dir + os.sep + 'senti_pos.txt', 'w') as ws, \
        open(output_dir + os.sep + 'senti_neg.txt', 'w') as ls:
            for item in term_senti_tuple:
                if item[1] != 0:
                    xs.write(item[0] + '\t' + str(item[1]*1000) + '\n')
                if item[1] > 0:
                    ws.write(item[0] + '\t' + str(item[1]*1000) + '\n')
                if item[1] < 0:
                    ls.write(item[0] + '\t' + str(item[1]*1000) + '\n')

