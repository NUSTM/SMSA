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

def gen_term_set(doc_terms_list, df = 1):
    term_set_dict = {}
    term_set = []
    for lst in doc_terms_list:
        tmp_lst = list(set(lst))
        for item in tmp_lst:
            if term_set_dict.has_key(item):
                term_set_dict[item] += 1
            else:
                term_set_dict[item] = 1
    for key, val in term_set_dict.iteritems():
        if val >= df:
            term_set.append(key)
    return term_set

def normalize_dict(input_dict):
    val_lst = input_dict.values()
    max_val = max(val_lst)
    min_val = min(val_lst)
    print max_val, min_val
    for key in input_dict:
        if input_dict[key] >= max_val:
            input_dict[key] = 1.0
        elif input_dict[key] <= min_val:
            input_dict[key] = -1.0
        else:
            new_val = -1 + (2 * (input_dict[key] - min_val)/(max_val - min_val))
            input_dict[key] = new_val


if __name__ == '__main__':
    ngram_list = ['uni']
    metric = 'DF'
    input_dir = '../sentiment-classification/corpus' + os.sep + 'zh_distant_data'

    pos_fenci_lines = [x.strip() for x in open(input_dir + os.sep + 'pos_fenci').readlines()]
    neg_fenci_lines = [x.strip() for x in open(input_dir + os.sep + 'neg_fenci').readlines()]

    for ngram in ngram_list:
        pos_terms_list = pytc.gen_N_gram(pos_fenci_lines, ngram)
        neg_terms_list = pytc.gen_N_gram(neg_fenci_lines, ngram)
        term_set = gen_term_set(pos_terms_list + neg_terms_list, df=1)

        term_pos_freq = {}.fromkeys(term_set, 0)
        term_neg_freq = {}.fromkeys(term_set, 0)

        freq_pos, freq_neg = 0, 0

        for lst in pos_terms_list:
            if metric == 'DF':
                lst = list(set(lst))
            for term in lst:
                if term in term_pos_freq:
                    term_pos_freq[term] += 1
            freq_pos += 1

        for lst in neg_terms_list:
            if metric == 'DF':
                lst = list(set(lst))
            for term in lst:
                if term in term_neg_freq:
                    term_neg_freq[term] += 1
            freq_neg += 1

        print "freq_pos=", freq_pos
        print "freq_neg=", freq_neg
        print "size of term_set", len(term_set)

        term_senti_dict = {}

        for term in term_set:
            if term in term_pos_freq and term in term_neg_freq:
                if term_pos_freq[term] == 0 and term_neg_freq[term] == 0:
                    term_senti_dict[term] = 0
                else:
                    tmp = ((term_pos_freq[term] + 1) * freq_neg) / ((term_neg_freq[term] + 1) * freq_pos)
                    term_senti_dict[term] = round(math.log(tmp, 2), 4)

        # normalize_dict(term_senti_dict)
        term_senti_tuple = sorted(term_senti_dict.iteritems(), key=lambda x:abs(x[1]), reverse=True)

        output_dir = 'DICT'

        with open(output_dir + os.sep + 'ZH_PMI.txt', 'w') as xs:
            for item in term_senti_tuple:
                if item[1] != 0:
                    xs.write(item[0] + '\t' + str(item[1]) + '\n')
