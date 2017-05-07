# -*- coding: utf-8 -*-
"""
@author: JieJ

一些文本操作的基本函数

"""

import os
import re

#########################################   文件读写工具  ######################################
def load_lexicon(fname, convert_func):
    '''加载文件为字典数据类型'''
    lst = [x.strip().split('\t') for x in open(fname).readlines()]
    word_dict = {}
    for l in lst:
        word_dict[l[0]] = convert_func(l[1])
    return word_dict

def store_lexicon(tar_dict, fname):
    '''加载文件为字典数据类型'''
    with open(fname, 'w') as xs:
        for key, val in tar_dict.iteritems():
            xs.write(str(key) +'\t' + str(val) + '\n')

def store_rule_result(result_dict_lst,info_names,output_dir):
    '''存储规则算法结果'''
    for info in info_names:
        f = open(output_dir+os.sep+info,'w')
        for res in result_dict_lst:
            f.write(str(res[info])+'\n')
        f.close()


def write_score_file(final_score_list, fname):
    '''规则得分写入文件'''
    f = open(fname,'w')
    f.writelines([str(x) + '\n' for x in final_score_list])
    f.close()

def write_score_file_2(index_score_list, fname):
    '''规则得分及对应序号写入文件'''
    with open(fname, 'w') as xs:
        for item in index_score_list:
            xs.write(str(item[0]) + '\t' + str(item[1]) + '\n')


#########################################   功能性函数  ######################################
def cal_len(doc):
    '''计算一篇文档的长度'''
    ss = ''.join(doc)
    ss = ss.replace(' ','').replace('\t','')
    ss = re.sub('\[.*?\]','',ss)
    return len(ss.decode('utf8','ignore'))

def cut_sentence(block,puncs_list):
    '''按照标点分割子句'''
    start = 0
    i = 0                                       #记录每个字符的位置
    sents = []
    ct = 0
    for word in block:
        if word in puncs_list:
            lst = block[start:i+1]
            if len(lst)==1:
                if ct>=1:
                    sents[ct-1].append(lst[0])
            else:
                sents.append(block[start:i+1])
                ct += 1
            start = i + 1                       #start标记到下一句的开头
            i += 1
        else:
            i += 1                              #若不是标点符号，则字符位置继续前移
    if start < len(block):
        sents.append(block[start:])             #这是为了处理文本末尾没有标点符号的情况
    return sents
