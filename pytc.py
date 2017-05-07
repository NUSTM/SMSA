# -*- coding: utf-8 -*-

from __future__ import division
import os
import re
import random
import math
import tools
import subprocess

from Lbsa import Lbsa

########### Global Parameters ###########

# TOOL_PATH = 'ML-TOOLS'
# LIBLINEAR_LEARN_EXE = TOOL_PATH + os.sep+ 'liblinear-1.96' + os.sep + 'windows' + os.sep + 'train.exe'
# LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + os.sep+ 'liblinear-1.96' + os.sep + 'windows' + os.sep + 'predict.exe'
# NB_LEARN_EXE = TOOL_PATH + os.sep+ 'openpr-nb_v1.16' + os.sep + 'windows' + os.sep + 'nb_learn.exe'
# NB_CLASSIFY_EXE = TOOL_PATH + os.sep+ 'openpr-nb_v1.16' + os.sep + 'windows' + os.sep + 'nb_classify.exe'
# LIBSVM_LEARN_EXE = TOOL_PATH + os.sep+ 'libsvm-3.2' + os.sep + 'windows' + os.sep + 'svm-train.exe'
# LIBSVM_CLASSIFY_EXE = TOOL_PATH + os.sep+ 'libsvm-3.2' + os.sep + 'windows' + os.sep + 'svm-predict.exe'

TOOL_PATH = '/home/poa/users/jj/sa_api'
# SVM_LEARN_EXE = TOOL_PATH + '\\svm_light\\svm_learn.exe'
# SVM_CLASSIFY_EXE = TOOL_PATH + '\\svm_light\\svm_classify.exe'

TOOL_PATH_2 = '/home/jjiang/SAS'

LIBLINEAR_LEARN_EXE = TOOL_PATH + '/liblinear-1.96/train'
LIBLINEAR_CLASSIFY_EXE = TOOL_PATH + '/liblinear-1.96/predict'
NB_LEARN_EXE = TOOL_PATH_2 + '/ML-TOOLS/openpr-nb_v1.16/nb_learn'
NB_CLASSIFY_EXE = TOOL_PATH_2 + '/ML-TOOLS/openpr-nb_v1.16/nb_classify'
NB_BIAS_LEARN_EXE = TOOL_PATH + '/wnb_linux/nbis_learn'
NB_BIAS_CLASSIFY_EXE = TOOL_PATH + '/wnb_linux/nbis_classify'
LIBSVM_LEARN_EXE = TOOL_PATH + '/libsvm-3.21/svm-train'
LIBSVM_CLASSIFY_EXE = TOOL_PATH + '/libsvm-3.21/svm-predict'
MALLET_EXE = '/home/jjiang/mallet-2.0.8/bin/mallet'
LOG_LIM = 1E-300

########## Data I/O Functions ##########

def read_annotated_data(fname_list, class_list):
    '''
    read data with class annotation, one class per file, one instance per line
    return instance list and corresponding class label list
    '''
    doc_str_list = []
    doc_class_list = []
    for doc_fname,class_fname in zip(fname_list, class_list):
        doc_str_list_one_class = [x.strip() for x in open(doc_fname, 'r').readlines()]
        doc_str_list.extend(doc_str_list_one_class)
        doc_class_list.extend([class_fname] * len(doc_str_list_one_class))
    return doc_str_list,doc_class_list

def read_unannotated_data(fname_list):
    '''
    read data without class annotation, one instance per line
    return instance list
    '''
    doc_str_list = []
    for doc_fname in fname_list:
        doc_str_list_one_class = [x.strip() for x in open(doc_fname, 'r').readlines()]
        doc_str_list.extend(doc_str_list_one_class)
    return doc_str_list


def gen_nfolds_f2(input_dir, output_dir, nfolds_num, fname_list, random_tag=False):
    '''
    Generate nfolds, with each fold containing a training fold and test fold
    '''
    class_dict = dict(zip(fname_list, [str(i) for i in range(1, len(fname_list) + 1)]))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    doc_str_test_dict, label_test_dict = dict(), dict()
    for fname in fname_list:
        doc_str_list = open(input_dir + os.sep + fname, 'r').readlines()
        # random.shuffle(doc_str_list)
        # if random_tag == True:
        #     random.shuffle(doc_str_list)
        doc_num = len(doc_str_list)
        pos_range = int(doc_num / nfolds_num)
        begin_pos = 0
        for fold_id in range(nfolds_num):
            fold_dir = output_dir + os.sep + 'fold' + str(fold_id+1)
            if not os.path.exists(fold_dir):
                os.mkdir(fold_dir)

            if fold_id != nfolds_num - 1:
                end_pos = begin_pos + pos_range
            else:
                end_pos = len(doc_str_list)

#            print fname,"fold"+str(fold_id+1),"begin_pos="+str(begin_pos),\
            "end_pos="+str(end_pos)

            doc_str_list_train = doc_str_list[:begin_pos] + doc_str_list[end_pos:]

            if doc_str_test_dict.has_key(fold_id):
                doc_str_test_dict[fold_id].extend(doc_str_list[begin_pos:end_pos])
            else:
                doc_str_test_dict[fold_id] = doc_str_list[begin_pos:end_pos]

            if label_test_dict.has_key(fold_id):
                label_test_dict[fold_id].extend([class_dict[fname]] * (end_pos - begin_pos))
            else:
                label_test_dict[fold_id] = [class_dict[fname]] * (end_pos - begin_pos)

            train_dir = fold_dir + os.sep + 'train'
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            fout = open(train_dir + os.sep + fname, 'w')
            fout.writelines([x for x in doc_str_list_train])
            fout.close()


            begin_pos = end_pos

    for fold_id in range(nfolds_num):
        fold_dir = output_dir + os.sep + 'fold' + str(fold_id+1)
        test_dir = fold_dir + os.sep + 'test'
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        if fname_list[0].endswith('fenci'):
            test_fname = 'test_fenci'
        if fname_list[0].endswith('pos'):
            test_fname = 'test_pos'
        with open(test_dir + os.sep + test_fname, 'w') as fs, \
        open(test_dir + os.sep + 'test_label', 'w') as ls:
            fs.writelines(doc_str_test_dict[fold_id])
            ls.writelines([x + '\n' for x in label_test_dict[fold_id]])

########## Feature Extraction Fuctions ##########

def get_doc_unis_list(doc_str_list):
    '''generate unigram language model for each segmented instance'''
    unis_list = [x.strip().split() for x in doc_str_list]
    return unis_list

def get_doc_bis_list(doc_str_list):
    '''generate bigram language model for each segmented instance'''
    unis_list = get_doc_unis_list(doc_str_list)
    doc_bis_list = []
    for k in range(len(doc_str_list)):
        unis = unis_list[k]
        if len(unis) == 0:
            doc_bis_list.append([])
            continue
        unis_pre, unis_after = ['<bos>'] + unis, unis + ['<eos>']
        doc_bis_list.append([x + '<w-w>' + y for x, y in zip(unis_pre, unis_after)])
    return doc_bis_list

def get_doc_triple_list(doc_str_list):
    '''generate triple-gram language model for each segmented instance'''
    doc_unis_list = get_doc_unis_list(doc_str_list)
    doc_bis_list = get_doc_bis_list(doc_str_list)
    doc_triple_list = []
    for k in range(len(doc_str_list)):
        unis = doc_unis_list[k]
        bis = doc_bis_list[k]
        if len(bis)<=2:
            doc_triple_list.append([])
            continue
        pre, after = bis[:-1], unis[1:] + ['<eos>']
        doc_triple_list.append([x + '<w-w>' + y for x, y in zip(pre, after)])
    return doc_triple_list

def get_doc_quat_list(doc_str_list):
    '''generate triple-gram language model for each segmented instance'''
    doc_unis_list = get_doc_unis_list(doc_str_list)
    doc_bis_list = get_doc_bis_list(doc_str_list)
    doc_triple_list = get_doc_triple_list(doc_str_list)
    doc_quat_list = []
    for k in range(len(doc_str_list)):
        unis = doc_unis_list[k]
        bis = doc_bis_list[k]
        triple = doc_triple_list[k]
        if len(triple)<=2:
            doc_quat_list.append([])
            continue
        pre, after = ['<bos>'] + unis[:-2], triple[1:]
        doc_quat_list.append([x+'<w-w>'+y for x,y in zip(pre,after)])
    return doc_quat_list

# def get_joint_sets(lst1, lst2):
#     '''
#     map corresponding element for two 2-dimention list
#     '''
#     if len(lst1) != len(lst2):
#         print "different lengths, return the first list object"
#         return lst1
#     return map(lambda x, y : x + y, lst1, lst2)

# def gen_N_gram(doc_str_list, ngram='uni'):
#     '''
#     generating NGRAM for each instance according to given N
#     '''
#     doc_ngram_list = []
#     if ngram=='uni':
#         doc_ngram_list = get_doc_unis_list(doc_str_list)
#     elif ngram=='bis':
#         doc_uni_list = get_doc_unis_list(doc_str_list)
#         doc_bis_list = get_doc_bis_list(doc_str_list)
#         doc_ngram_list = get_joint_sets(doc_uni_list, doc_bis_list)
#     elif ngram=='tri':
#         doc_uni_list = get_doc_unis_list(doc_str_list)
#         doc_bis_list = get_doc_bis_list(doc_str_list)
#         doc_trip_list = get_doc_triple_list(doc_str_list)
#         tmp = get_joint_sets(doc_uni_list, doc_bis_list)
#         doc_ngram_list = get_joint_sets(tmp,doc_trip_list)
#     elif ngram=='quat':
#         doc_uni_list = get_doc_unis_list(doc_str_list)
#         doc_bis_list = get_doc_bis_list(doc_str_list)
#         doc_trip_list = get_doc_triple_list(doc_str_list)
#         doc_quat_list = get_doc_quat_list(doc_str_list)
#         tmp1 = get_joint_sets(doc_uni_list, doc_bis_list)
#         tmp2 = get_joint_sets(tmp1, doc_trip_list)
#         doc_ngram_list = get_joint_sets(tmp2,doc_quat_list)
#     else:
#         for i in range(len(doc_str_list)):
#             doc_ngram_list.append([])
#     return doc_ngram_list

def get_joint_sets(lst1, lst2):
    '''
    map corresponding element for two 2-dimention list
    '''
    if len(lst1) == 0 or len(lst1) != len(lst2):
        print "different lengths, do nothing"
        return
    for sub_lst1, sub_lst2 in zip(lst1, lst2):
        sub_lst1.extend(sub_lst2)

def gen_N_gram(doc_str_list, ngram='uni'):
    '''
    generating NGRAM for each instance according to given N
    '''
    if ngram=='uni':
        doc_uni_list = get_doc_unis_list(doc_str_list)
        return doc_uni_list
    elif ngram=='bis':
        doc_bis_list = get_doc_bis_list(doc_str_list)
        return doc_bis_list
    elif ngram=='tri':
        doc_trip_list = get_doc_triple_list(doc_str_list)
        return doc_trip_list
    elif ngram=='quat':
        doc_quat_list = get_doc_quat_list(doc_str_list)
        return doc_quat_list
    else:
        doc_ngram_list = []
        for i in range(len(doc_str_list)):
            doc_ngram_list.append([])
        return doc_ngram_list

def gen_character_ngram_list(doc_str_list, ngram):
    doc_terms_list = []
    ngram_dict = {
        'uni' : 1,
        'bis': 2,
        'tri': 3,
        'quat': 4,
        'five': 5,
        'six': 6,
    }
    if not ngram_dict.has_key(ngram):
        print "ngram key error"
        return doc_terms_list

    l =ngram_dict[ngram]

    if l <= 1:
        print "character ngram too short"
        return doc_terms_list

    for doc in doc_str_list:
        doc_str = doc.replace(' ','')
        doc_str = re.sub('\[.*?\]', '', doc_str)
        doc_str = doc_str.decode('utf8', 'ignore')
        terms_lst = []
        for i in range(len(doc_str)):
            term = doc_str[i: i + l].encode('utf8', 'ignore')
            if re.search('[\(\)\.;,/，。；《》【】]', term) is not None:
                continue
            term += '_c#'
            terms_lst.append(term)
        terms_lst = list(set(terms_lst))
        doc_terms_list.append(terms_lst)
    return doc_terms_list

# def gen_character_ngram_list(doc_str_list, ngram):
#     doc_terms_list = []
#     if ngram <= 1:
#         return doc_terms_list

#     for doc in doc_str_list:
#         doc_str = doc.replace(' ','')
#         doc_str = re.sub('\[.*?\]', '', doc_str)
#         doc_str = doc_str.decode('utf8', 'ignore')
#         terms_lst = []
#         for i in range(len(doc_str)):
#             for j in range(2,5):
#                 term = doc_str[i:i+j].encode('utf8', 'ignore')
#                 if re.search('[\(\)\.;,/，。；《》【】]', term) is not None:
#                     continue
#                 terms_lst.append(term)
#         terms_lst = list(set(terms_lst))
#         doc_terms_list.append(terms_lst)
#     return doc_terms_list

def get_term_set(doc_terms_list):
    '''generate unique term set fron N segmented instances, N = len(doc_terms_list) '''
    term_set = set()
    for doc_terms in doc_terms_list:
        term_set.update(doc_terms)
    return sorted(list(term_set))

def save_term_set(term_set, fname):
    '''save term set'''
    open(fname, 'w').writelines([x + '\n' for x in term_set])

def load_term_set(fname):
    '''load term set'''
    term_set = [x.strip() for x in open(fname, 'r').readlines()]
    return term_set

def stat_df_class(class_list, doc_class_list):
    '''calculate df num for each class label'''
    df_class = [doc_class_list.count(x) for x in class_list]
    return df_class

def stat_df_term(term_set, doc_terms_list):
    '''calculate df num for each term'''
    df_term = {}.fromkeys(term_set, 0)
    for doc_terms in doc_terms_list:
        for term in set(doc_terms):
            if df_term.has_key(term):
                df_term[term] += 1
    return df_term

def stat_df_term_class(term_set, class_list, doc_terms_list, doc_class_list):
    '''calculate df num in every class label for each term'''
    class_id_dict = dict(zip(class_list, range(len(class_list))))
    df_term_class = {}
    for term in term_set:
        df_term_class[term] = [0]*len(class_list)
    for k in range(len(doc_class_list)):
        class_label = doc_class_list[k]
        class_id = class_id_dict[class_label]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            if df_term_class.has_key(term):
                df_term_class[term][class_id] += 1
    return df_term_class

########## Feature Selection Fuctions ##########

def feature_selection_df(df_term, df_num):
    term_set_df = []
    for term in sorted(df_term.keys()):
        if df_term[term] >= df_num:
            term_set_df.append(term)
    return term_set_df

def supervised_feature_selection(df_class, df_term_class, fs_method='IG',
                                 fs_num=0, fs_class=-1):
    if fs_method == 'MI':
        term_set_fs, term_score_dict = feature_selection_mi(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'IG':
        term_set_fs, term_score_dict = feature_selection_ig(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'CHI':
        term_set_fs, term_score_dict = feature_selection_chi(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'WLLR':
        term_set_fs, term_score_dict = feature_selection_wllr(df_class, \
            df_term_class, fs_num, fs_class)
    elif fs_method == 'LLR':
        term_set_fs, term_score_dict = feature_selection_llr(df_class, \
            df_term_class, fs_num, fs_class)
    return term_set_fs, term_score_dict

def feature_selection_mi(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
            p_c = float(cap_a+cap_c) / cap_n
            score = math.log(p_c_t / p_c)
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

def feature_selection_ig(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            p_c = float(cap_a + cap_c) / cap_n
            p_t = float(cap_a + cap_b) / cap_n
            p_nt = 1 - p_t
            p_c_t = (cap_a + 1.0) / (cap_a + cap_b + class_set_size)
            p_c_nt = (cap_c + 1.0) / (cap_c + cap_d + class_set_size)
            score = - p_c * math.log(p_c) + p_t * p_c_t * math.log(p_c_t) + \
                p_nt * p_c_nt * math.log(p_c_nt)
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


def feature_selection_chi(df_class, df_term_class, fs_num=0, fs_class=-1):
    #df_class:不同类别的文档数
    #df_term_class: 每个词在各个类别下的文档数
    #{term1:[class1:num,class2:num,..],..}
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            cap_d = cap_n - cap_a - cap_c - cap_b
            cap_nu = float(cap_a * cap_d - cap_c * cap_b)
            cap_x1 = cap_nu / ((cap_a + cap_c) * (cap_b + cap_d))
            cap_x2 = cap_nu / ((cap_a + cap_b) * (cap_c + cap_d))
            score = cap_n * cap_x1 * cap_x2
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

def feature_selection_llr(df_class, df_term_class, fs_num=0, fs_class=-1):
    term_set = df_term_class.keys()
    term_score_dict = {}.fromkeys(term_set)
    for term in term_set:
        df_list = df_term_class[term]
        class_set_size = len(df_list)
        cap_n = sum(df_class)
        score_list = []
        for class_id in range(class_set_size):
            cap_a = df_list[class_id]
            cap_b = sum(df_list) - cap_a
            cap_c = df_class[class_id] - cap_a
            p_c_t = (cap_a + 1.0)/(cap_a + cap_b + class_set_size)
            p_nc_t = 1 - p_c_t
            p_c = float(cap_a + cap_c)/ cap_n
            p_nc = 1 - p_c
            score = math.log(p_c_t * p_nc / (p_nc_t * p_c))
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

def feature_selection_all(doc_terms_list, doc_class_list, class_fname_list,
    term_set, fs_opt, fs_df_num, fs_method, fs_num):
    import copy
    term_set_train = copy.deepcopy(term_set)

    #是否进行监督方式的特征选择
    if fs_opt == 1:
        # 如果fs_num == -1, 则默认为选择一半数量的特征词
        if fs_num==-1:
            fs_num = int(len(term_set)/2)

        print 'Selecting features using', fs_method, 'method ......'
        df_class = stat_df_class(class_fname_list, doc_class_list)
        df_term_class = stat_df_term_class(term_set_train, class_fname_list, \
            doc_terms_list, doc_class_list)
        term_set_fs, term_score_dict = supervised_feature_selection(df_class, \
            df_term_class, fs_method, fs_num, fs_class=-1)
        term_set_train = term_set_fs
        # print 'after supervised fs, Feature Num:', len(term_set_train)

    #是否进行基于文档频率的特征选择
    if fs_df_num >= 2:
        print 'Filtering features DF>=',fs_df_num,'...'
        term_df = stat_df_term(term_set_train, doc_terms_list)
        term_set_df = feature_selection_df(term_df, fs_df_num)
        term_set_train = term_set_df
        # print 'after df fs, Feature Num:', len(term_set_train)

    print 'final Feature Num:', len(term_set_train)
    return term_set_train

########## Building Sample Files ##########

def switch_final_score(final_score):
    '''分值转换'''
    if final_score>0:
        final_score = 3
    elif final_score<0:
        final_score = 1
    else:
        final_score = 2
    return final_score

def trans_feature_weight(weight1,weight2,fixed_id,samp_dict):
    '''将离散值转化为BOOL值'''
    if(weight1<weight2):
        samp_dict[fixed_id] = 1
    elif(weight1>weight2):
        samp_dict[fixed_id+1] = 1
    else:
        samp_dict[fixed_id+2] = 1
    fixed_id += 3
    return fixed_id




def lexicon_score(doc, tmp_dict):
    score = 0
    doc = list(set(doc))
    for term in doc:
        if term in tmp_dict:
            score += tmp_dict[term]
    return score

def build_samps(term_dict, class_dict, doc_class_list, doc_terms_list, doc_uni_token,
    term_weight, rule_feature=0, idf_term=[], embeddings = []):
    samp_dict_list = []
    samp_class_list = []

    for k in range(len(doc_class_list)):
        doc_class = doc_class_list[k]
        samp_class = class_dict[doc_class]
        samp_class_list.append(samp_class)

        doc_terms = doc_terms_list[k]
        samp_dict = {}

        if rule_feature != 100:
            for term in doc_terms:
                if term_dict.has_key(term):
                    term_id = term_dict[term]
                    if term_weight == 'BOOL':
                        samp_dict[term_id] = 1
                    elif term_weight == 'TF':
                        if samp_dict.has_key(term_id):
                            samp_dict[term_id] += 1
                        else:
                            samp_dict[term_id] = 1
                    elif term_weight == 'TFIDF':
                        if samp_dict.has_key(term_id):
                            samp_dict[term_id] += idf_term[term]
                        else:
                            samp_dict[term_id] = idf_term[term]

            fixed_id = len(term_dict)+1        #下一个特征开始的ID号
        else:
            fixed_id = 1

        if rule_feature != 0:

            if rule_feature >=1:
                if len(embeddings) > 0:
                    emb = embeddings[k]
                    for val in emb:
                        if val != 0:
                            samp_dict[fixed_id] = val
                        fixed_id += 1

            # # 初始化情感分析对象, 该对象使用带有强度标注的情感词典进行分析
            # win_size = 4
            # phrase_size = 3
            # test = Lbsa(win_size,phrase_size)
            # doc_tokens = doc_uni_token[k]

            # end_num = rule_feature
            # sc_lst = test.lexicon_score_index(doc_tokens, end_num)
            # for sc in sc_lst:
            #     if sc != 0:
            #         samp_dict[fixed_id] = sc
            #     fixed_id += 1

            # rule_score = 0
            # if rule_feature%5 == 1:
            #     rule_score = test.en_pmi_distant_dict_score(doc_tokens)
            # elif rule_feature%5 == 2:
            #     rule_score = test.en_nn_distant_dict_score(doc_tokens)
            # elif rule_feature%5 == 3:
            #     rule_score = test.zh_pmi_distant_dict_score(doc_tokens)
            # elif rule_feature%5 == 4:
            #     rule_score = test.zh_nn_distant_dict_score(doc_tokens)

            # if rule_score != 0:
            #     samp_dict[fixed_id] = rule_score
            # fixed_id += 1

        samp_dict_list.append(samp_dict)
    return samp_dict_list, samp_class_list

# def build_samps(term_dict, class_dict, doc_class_list, doc_terms_list, doc_uni_token,
#     term_weight, rule_feature=0, senti_dict_lst=[], embeddings = []):
#     samp_dict_list = []
#     samp_class_list = []

#     for k in range(len(doc_class_list)):
#         doc_class = doc_class_list[k]
#         samp_class = class_dict[doc_class]
#         samp_class_list.append(samp_class)

#         doc_terms = doc_terms_list[k]
#         samp_dict = {}

#         if rule_feature != 10:
#             for term in doc_terms:
#                 if term_dict.has_key(term):
#                     term_id = term_dict[term]
#                     if term_weight == 'BOOL':
#                         samp_dict[term_id] = 1
#                     elif term_weight == 'TF':
#                         if samp_dict.has_key(term_id):
#                             samp_dict[term_id] += 1
#                         else:
#                             samp_dict[term_id] = 1
#                     elif term_weight == 'TFIDF':
#                         if samp_dict.has_key(term_id):
#                             samp_dict[term_id] += idf_term[term]
#                         else:
#                             samp_dict[term_id] = idf_term[term]

#             fixed_id = len(term_dict)+1        #下一个特征开始的ID号
#         else:
#             fixed_id = 1

#         if rule_feature != 0:

#             # 初始化情感分析对象, 该对象使用带有强度标注的情感词典进行分析
#             win_size = 4
#             phrase_size = 3
#             test = Lbsa(win_size,phrase_size)
#             doc_tokens = doc_uni_token[k]

#             #######               添加规则情感特征                 ###########
#             # 添加情感词相关特征
#             rule_result = test.cal_document(doc_tokens,'none')
#             for key in ['final_score']:
#                 val = rule_result[key]
#                 if float(val) !=0:
#                     samp_dict[fixed_id] = float(val)
#                 fixed_id += 1

#             rule_score = test.zh_pmi_distant_dict_score(doc_tokens)

#             if rule_score != 0:
#                 samp_dict[fixed_id] = rule_score
#             fixed_id += 1

#             # # 添加语义特征，包括： 否定词数量，程度副词数量，感叹词数量，第一人称词数量，第二人称词数量
#             # for key in ['deny_ct','degree_ct']:
#             #     val = rule_result[key]
#             #     if float(val) !=0:
#             #         samp_dict[fixed_id] = float(val)
#             #     fixed_id += 1

#             yuqici_ct = doc_tokens.count('啊')+doc_tokens.count('啦')+doc_tokens.count('呀')+\
#             doc_tokens.count('吧')+doc_tokens.count('哇')+doc_tokens.count('哦')
#             if yuqici_ct > 0:
#                 samp_dict[fixed_id] = float(yuqici_ct)
#             fixed_id += 1

#             if '我' in doc_terms or '我们' in doc_terms:
#                 samp_dict[fixed_id] = 1
#             fixed_id += 1
#             if '你' in doc_terms or '你们' in doc_terms:
#                 samp_dict[fixed_id] = 1
#             fixed_id += 1

#             # # 标点符号特征
#             # for punc in ['!','?','！','！！！','？','？？？','。。。']:
#             # # for punc in ['?','？']:
#             #     punc_ct = doc_tokens.count(punc)
#             #     if punc_ct > 0:
#             #         samp_dict[fixed_id] = float(punc_ct)
#             #     fixed_id += 1




#             # for key in ['pos_ct','neg_ct']:
#             #     val = rule_result[key]
#             #     if float(val) !=0:
#             #         samp_dict[fixed_id] = float(val)
#             #     fixed_id += 1

#             # for key in ['pos_sub','neg_sub','sub_ct']:
#             #     val = rule_result[key]
#             #     if float(val) !=0:
#             #         samp_dict[fixed_id] = float(val)
#             #     fixed_id += 1


#             score = switch_final_score(rule_result['final_score'])
#             fixed_id = trans_feature_weight(score,2,fixed_id,samp_dict)

#             # face_score = switch_final_score(rule_result['face_score'])
#             # fixed_id = trans_feature_weight(face_score,2,fixed_id,samp_dict)

#             # regular_score = switch_final_score(rule_result['final_score'] - rule_result['face_score'])
#             # fixed_id = trans_feature_weight(regular_score,2,fixed_id,samp_dict)

#             #正向情感词数量是否多于消极情感词
#             fixed_id = trans_feature_weight(rule_result['pos_ct'],rule_result['neg_ct'],fixed_id,samp_dict)

#             # 正向子句数是否多于负向子句数
#             fixed_id = trans_feature_weight(rule_result['pos_sub'],rule_result['neg_sub'],fixed_id,samp_dict)




#             # for senti_dict in senti_dict_lst:
#             #     # pos_word_num, neg_word_num = test.general_lex_method(doc_tokens, senti_dict, 4)
#             #     # pos_word_num, neg_word_num = test.character_ngram_method(doc_tokens, senti_dict, 4)
#             #     pos_word_num, neg_word_num, your_score = test.your_dict_score(doc_tokens, senti_dict)
#             #     # print test.your_dict_score(doc_tokens, senti_dict)
#             #     if pos_word_num > 0:
#             #         samp_dict[fixed_id] = pos_word_num
#             #     fixed_id += 1
#             #     if neg_word_num > 0:
#             #         samp_dict[fixed_id] = neg_word_num
#             #     fixed_id += 1
#             #     # fixed_id = trans_feature_weight(pos_word_num, neg_word_num, fixed_id, samp_dict)
#             #     your_score = switch_final_score(your_score)
#             #     fixed_id = trans_feature_weight(your_score, 2, fixed_id, samp_dict)
#         #########################################################################
#         samp_dict_list.append(samp_dict)
#     return samp_dict_list, samp_class_list

def save_samps(samp_dict_list, samp_class_list, fname, feat_num=0):
    length = len(samp_class_list)
    fout = open(fname, 'w')
    for k in range(length):
        samp_dict = samp_dict_list[k]
        samp_class = samp_class_list[k]
        if samp_class == 0:
            samp_class = 1
        fout.write(str(samp_class) + '\t')
        # fout.write(str(samp_class) + ' ')
        for term_id in sorted(samp_dict.keys()):
            if feat_num == 0 or term_id < feat_num:
                fout.write(str(term_id) + ':' + str(samp_dict[term_id]) + ' ')
        fout.write('\n')
    fout.close()

def save_weights(samp_weight_list, fname_samp_weight):
    with open(fname_samp_weight, 'w') as xs:
        xs.writelines([x + '\n' for x in samp_weight_list])

########## Learning & Classification Functions Using Machine Learning Tools##########

def ml_learn_classify(classifier_name, train_samp_dir, model_file_dir, test_samp_dir, result_file_dir):

    if classifier_name == 'lg':
        liblinear_learn(train_samp_dir, model_file_dir)
        liblinear_predict(test_samp_dir, model_file_dir, result_file_dir)
    elif classifier_name == 'nb':
        nb_learn(train_samp_dir, model_file_dir)
        nb_predict(test_samp_dir, model_file_dir, result_file_dir)
    elif classifier_name == 'svm':
        libsvm_learn(train_samp_dir, model_file_dir)
        libsvm_predict(test_samp_dir, model_file_dir, result_file_dir)
    elif classifier_name == 'maxent':
        maxent_learn(train_samp_dir, model_file_dir)
        maxent_predict(test_samp_dir, model_file_dir, result_file_dir)

    else:
        pass


def maxent_learn(train_samp_dir, model_file_dir, model_fname = None):
    fname_samp_train = train_samp_dir + os.sep + 'train.samp'
    fname_mallet_samp_train = train_samp_dir + os.sep + 'train.mallet'
    fname_model = model_file_dir + os.sep + 'maxent.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)

    import subprocess
    pop = subprocess.Popen(MALLET_EXE + ' import-svmlight --input ' + \
        fname_samp_train + ' --output ' + fname_mallet_samp_train, shell=True)
    pop.wait()

    pop = subprocess.Popen(MALLET_EXE + ' train-classifier --input ' + \
        fname_mallet_samp_train + ' --trainer MaxEnt --output-classifier ' + fname_model, shell=True)
    pop.wait()

def maxent_predict(test_samp_dir, model_file_dir, result_file_dir, model_fname = None,result_fname = None):
    fname_samp_test = test_samp_dir + os.sep + 'test.samp'
    fname_mallet_samp_test = test_samp_dir + os.sep + 'test.mallet'
    fname_model = model_file_dir + os.sep + 'maxent.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)
    fname_result = result_file_dir + os.sep + 'maxent.result'
    if result_fname is not None:
        fname_result = result_file_dir + os.sep + str(result_fname)

    import subprocess
    # pop = subprocess.Popen(MALLET_EXE + ' import-svmlight --input ' + \
    #     fname_samp_test + ' --output ' + fname_mallet_samp_test, shell=True)
    # pop.wait()

    pop = subprocess.Popen(MALLET_EXE + ' classify-svmlight --input ' + \
        fname_samp_test + ' --output  - --classifier '+ fname_model + ' > ' + \
        fname_result, shell=True)
    pop.wait()

    # convert to liblinear result file
    f_res = open(fname_result, 'r')
    res_lst = f_res.readlines()
    f_res.close()

    with open(fname_result, 'w') as xs:
        xs.write('labels 1 2 3\n')
        for line in res_lst:
            lst = line.strip().split('\t')
            proba = []
            for i in range(1, len(lst)):
                if i%2==0:
                    proba.append(float(lst[i]))
            import numpy as np
            prob_arr = np.asarray(proba)
            class_label = prob_arr.argmax(axis = 0)
            ss = ' '.join([str(x) for x in proba])
            xs.write(str(class_label+1)+ ' ')
            xs.write(ss + '\n')
            # if float(lst[2]) >= float(lst[4]):
            #     xs.write('1 ' + lst[2] + ' ' + lst[4] + '\n')
            # else:
            #     xs.write('2 ' + lst[2] + ' ' + lst[4] + '\n')

def liblinear_learn(train_samp_dir, model_file_dir, learn_opt='-s 7 -c 1', model_fname = None):
    fname_samp_train = train_samp_dir + os.sep + 'train.samp'
    fname_model = model_file_dir + os.sep + 'lg.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)

    import subprocess
    pop = subprocess.Popen(LIBLINEAR_LEARN_EXE + ' ' +  learn_opt + ' ' + \
    fname_samp_train + ' ' + fname_model, shell=True)
    pop.wait()

def liblinear_predict(test_samp_dir, model_file_dir, result_file_dir,
    classify_opt='-b 1', model_fname = None,result_fname = None):
    fname_model = model_file_dir + os.sep + 'lg.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)
    fname_samp_test = test_samp_dir + os.sep + 'test.samp'
    # fname_samp_test = test_samp_dir + os.sep + 'train.samp'
    fname_result = result_file_dir + os.sep + 'lg.result'
    if result_fname is not None:
        fname_result = result_file_dir + os.sep + str(result_fname)

    pop = subprocess.Popen(LIBLINEAR_CLASSIFY_EXE + ' ' + classify_opt + ' ' \
            + fname_samp_test + ' ' + fname_model + ' ' + fname_result, shell=True)
    pop.wait()

def nb_learn(train_samp_dir, model_file_dir, learn_opt='-e 1', model_fname = None):
    fname_samp_train = train_samp_dir + os.sep + 'train.samp'
    fname_model = model_file_dir + os.sep + 'nb.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)

    pop = subprocess.Popen(NB_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model, shell=True)
    pop.wait()

def nb_predict(test_samp_dir, model_file_dir, result_file_dir, classify_opt='-f 2',
    model_fname = None,result_fname = None):
    fname_model =model_file_dir + os.sep + 'nb.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)

    fname_samp_test = test_samp_dir + os.sep + 'test.samp'
    fname_result = result_file_dir + os.sep + 'nb.result'
    if result_fname is not None:
        fname_result = result_file_dir + os.sep + str(result_fname)

    pop = subprocess.Popen(NB_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_result, shell=True)
    pop.wait()

def libsvm_learn(train_samp_dir, model_file_dir, learn_opt='-t 0 -c 1 -b 1', model_fname = None):
    fname_samp_train = train_samp_dir + os.sep + 'train.samp'
    fname_model = model_file_dir + os.sep + 'svm.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)
    pop = subprocess.Popen(LIBSVM_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_model, shell=True)
    pop.wait()

def libsvm_predict(test_samp_dir, model_file_dir, result_file_dir, classify_opt='-b 1',
    model_fname = None,result_fname = None):
    fname_model =model_file_dir + os.sep + 'svm.model'
    if model_fname is not None:
        fname_model = model_file_dir + os.sep + str(model_fname)
    fname_samp_test = test_samp_dir + os.sep + 'test.samp'
    fname_result = result_file_dir + os.sep + 'svm.result'
    if result_fname is not None:
        fname_result = result_file_dir + os.sep + str(result_fname)

    pop = subprocess.Popen(LIBSVM_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_result, shell=True)
    pop.wait()


def nbis_learn(train_samp_dir, model_file_dir, learn_opt='-e 1'):
    fname_samp_train = train_samp_dir + os.sep + 'train.samp'
    fname_samp_weight = train_samp_dir + os.sep + 'train.weight'
    fname_model = model_file_dir + os.sep + 'nbis.model'

    pop = subprocess.Popen(NB_BIAS_LEARN_EXE + ' ' +  learn_opt + ' ' + \
        fname_samp_train + ' ' + fname_samp_weight + ' ' + fname_model, shell=True)
    pop.wait()

def nbis_predict(test_samp_dir, model_file_dir, result_file_dir, classify_opt='-f 2'):
    fname_model =model_file_dir + os.sep + 'nbis.model'
    fname_samp_test = test_samp_dir + os.sep + 'test.samp'
    fname_result = result_file_dir + os.sep + 'nbis.result'

    pop = subprocess.Popen(NB_BIAS_CLASSIFY_EXE + ' ' + classify_opt + ' ' + \
        fname_samp_test + ' ' + fname_model + ' ' + fname_result, shell=True)
    pop.wait()





def mix_model(output_dir, classifier_list):
    import numpy as np
    prob_mat_lst = []
    for classifier in classifier_list:
        f_output = open(output_dir + os.sep + classifier + '.result')
        output = f_output.readlines()
        f_output.close()
        start = 0
        if classifier == 'lg' or classifier == 'svm':
            start += 1
        prob_mat = []
        for i in range(start, len(output)):
            tmp = output[i].strip().split()[1:]
            prob_mat.append([float(tmp[j]) for j in range(len(tmp))])
        prob_mat = np.asarray(prob_mat)
        prob_mat_lst.append(prob_mat)

    #初始化一个0概率矩阵，并叠加所有分类器的概率，求平均值
    mix_prob_mat = np.asarray(np.zeros(prob_mat_lst[0].shape))
    for prob_mat in prob_mat_lst:
        print prob_mat.shape, mix_prob_mat.shape
        mix_prob_mat = mix_prob_mat + prob_mat
    mix_prob_mat =  mix_prob_mat / len(prob_mat_lst)  #求平均概率

    #筛选出每行最大概率的index，即为最终predict index
    max_prob_label = np.argmax(mix_prob_mat, 1)  #选出融合概率矩阵中每行最大的

    #将融合结果标签和相应概率写入文件
    f_mixed = open(output_dir + os.sep + 'mixed.result','w')
    for i in range(len(max_prob_label)):
        l = str(max_prob_label[i] + 1)
        ss = ''
        for k in range(len(mix_prob_mat[0])):
            ss += str(round(mix_prob_mat[i][k], 6)) + ' '
        f_mixed.write(l + '\t' + ss.rstrip())
        if i!=len(max_prob_label) - 1:
            f_mixed.write('\n')
    f_mixed.close()
    # pred_list_mix = [str(max_prob_label[i] + 1) for i in range(len(max_prob_label))]
    # return pred_list_mix

# def mix_model(self, test_dir, model_file_dir, mix_classifier_list):
#     pred_list_mix = pytc.mix_classifier(output_dir,mix_classifier_list)
#     class_list_test = [x.strip() for x in open(test_dir + os.sep + 'test_label').readlines()]
#     print len(pred_list_mix),len(class_list_test)
#     # mix_acc = pytc.calc_acc(pred_list_mix, class_list_test)
#     # print "mix_acc:",mix_acc
#     return mix_acc
