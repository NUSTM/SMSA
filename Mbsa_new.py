# -*- coding: utf-8 -*-
"""
@date: 2016.5.9
@author: JieJ

"""
import os
import random
import params
import pytc
from performance import performance


class Mbsa(object):
    def __init__(self, ngram_dict, term_weight='BOOL', rule_feature=0):
        self.token_gram = ngram_dict['token_gram']
        self.pos_gram = ngram_dict['pos_gram']
        self.tag_gram = ngram_dict['tag_gram']
        self.character_gram = ngram_dict['character_gram']
        self.term_weight = term_weight
        self.rule_feature = rule_feature

    def gen_doc_terms_list(self, samp_params, train_opt = 0):
        doc_class_list = []

        # token ngram
        doc_token_list, token_set = {}, {}
        doc_pos_list, pos_set = {}, {}
        doc_tag_list, tag_set = {}, {}
        doc_character_list, character_set = {}, {}


        doc_str_token, doc_class_list = pytc.read_annotated_data([samp_params['raw_dir'] + \
            os.sep + x for x in samp_params['token']], samp_params['class_name'])

        if len(self.pos_gram.keys()) > 0:
            doc_str_pos, doc_class_list = pytc.read_annotated_data([samp_params['raw_dir'] + os.sep + x \
                for x in samp_params['pos']], samp_params['class_name'])

        if len(self.tag_gram.keys()) > 0:
            doc_str_tag, doc_class_list = pytc.read_annotated_data([samp_params['raw_dir'] + os.sep + x \
                for x in samp_params['tag']], samp_params['class_name'])

        for gram_key in ['uni', 'bis', 'tri', 'quat', 'five', 'six']:
            if self.token_gram.has_key(gram_key):
                doc_token_list[gram_key] = pytc.gen_N_gram(doc_str_token, gram_key)
                token_set[gram_key] = pytc.get_term_set(doc_token_list[gram_key])
                params = self.token_gram[gram_key]

                # if it's in training producure
                if train_opt == 1 and len(doc_token_list[gram_key]) > 0:
                    print "token " + gram_key, "len(token_set)=",len(token_set[gram_key])
                    token_set[gram_key] = pytc.feature_selection_all(doc_token_list[gram_key], doc_class_list, samp_params['class_name'],
                        token_set[gram_key], params['fs_opt'], params['df'], params['fs_method'], params['fs_num'])
                    print "token " + gram_key, "after feature selection, len(token_set) =", len(token_set[gram_key])

            if self.pos_gram.has_key(gram_key):
                doc_pos_list[gram_key] = pytc.gen_N_gram(doc_str_pos, gram_key)
                pos_set[gram_key] = pytc.get_term_set(doc_pos_list[gram_key])
                params = self.pos_gram[gram_key]
                # if it's in training producure
                if train_opt == 1 and len(doc_pos_list[gram_key]) > 0:
                    print "pos " + gram_key, "len(pos_set)=",len(pos_set[gram_key])
                    pos_set[gram_key] = pytc.feature_selection_all(doc_pos_list[gram_key], doc_class_list, samp_params['class_name'],
                        pos_set[gram_key], params['fs_opt'], params['df'], params['fs_method'], params['fs_num'])
                    print "pos " + gram_key, "after feature selection, len(pos_set) =", len(pos_set[gram_key])

            if self.tag_gram.has_key(gram_key):
                doc_tag_list[gram_key] = pytc.gen_N_gram(doc_str_tag, gram_key)
                tag_set[gram_key] = pytc.get_term_set(doc_tag_list[gram_key])
                params = self.tag_gram[gram_key]

                # if it's in training producure
                if train_opt == 1 and len(doc_tag_list[gram_key]) > 0:
                    print "tag " + gram_key, "len(tag_set)=",len(tag_set[gram_key])
                    tag_set[gram_key] = pytc.feature_selection_all(doc_tag_list[gram_key], doc_class_list, samp_params['class_name'],
                        tag_set[gram_key], params['fs_opt'], params['df'], params['fs_method'], params['fs_num'])
                    print "tag " + gram_key, "after feature selection, len(tag_set) =", len(tag_set[gram_key])

            if self.character_gram.has_key(gram_key):
                doc_character_list[gram_key] = pytc.gen_character_ngram_list(doc_str_token, gram_key)
                character_set[gram_key] = pytc.get_term_set(doc_character_list[gram_key])
                params = self.character_gram[gram_key]

                # if it's in training producure
                if train_opt == 1 and len(doc_character_list[gram_key]) > 0:
                    print "character " + gram_key, "len(character_set)=",len(character_set[gram_key])
                    character_set[gram_key] = pytc.feature_selection_all(doc_character_list[gram_key], doc_class_list, samp_params['class_name'],
                        character_set[gram_key], params['fs_opt'], params['df'], params['fs_method'], params['fs_num'])
                    print "character " + gram_key, "after feature selection, len(character_set) =", len(character_set[gram_key])

        doc_terms_list, term_set = [], []
        for gram_key in ['uni', 'bis', 'tri', 'quat', 'five', 'six']:
            if self.token_gram.has_key(gram_key):
                term_set += token_set[gram_key]
                if len(doc_terms_list) == 0:
                    doc_terms_list = doc_token_list[gram_key]
                else:
                    pytc.get_joint_sets(doc_terms_list, doc_token_list[gram_key])

            if self.pos_gram.has_key(gram_key):
                term_set += pos_set[gram_key]
                if len(doc_terms_list) == 0:
                    doc_terms_list = doc_pos_list[gram_key]
                else:
                    pytc.get_joint_sets(doc_terms_list, doc_pos_list[gram_key])

            if self.tag_gram.has_key(gram_key):
                term_set += tag_set[gram_key]
                if len(doc_terms_list) == 0:
                    doc_terms_list = doc_tag_list[gram_key]
                else:
                    pytc.get_joint_sets(doc_terms_list, doc_tag_list[gram_key])

            if self.character_gram.has_key(gram_key):
                term_set += character_set[gram_key]
                if len(doc_terms_list) == 0:
                    doc_terms_list = doc_character_list[gram_key]
                else:
                    pytc.get_joint_sets(doc_terms_list, doc_character_list[gram_key])


        return doc_class_list, doc_str_token, doc_terms_list, term_set

    def gen_train_samps(self, train_samp_params):
        fname_samps_train = train_samp_params['samp_dir'] + os.sep + 'train.samp'
        fname_term_set_train = train_samp_params['term_set_dir'] + os.sep + 'term.set'

        doc_class_list_train, doc_str_token_train, doc_terms_list_train, term_set_train = \
        self.gen_doc_terms_list(train_samp_params, train_opt = 1)
        pytc.save_term_set(term_set_train, fname_term_set_train)

        # unigram的token，单独作为参数用来构建其他规则特征
        doc_uni_token_train = pytc.gen_N_gram(doc_str_token_train,'uni')

        term_dict = dict(zip(term_set_train, range(1,len(term_set_train)+1)))
        class_dict = dict(zip(train_samp_params['class_name'], range(1,1+len(train_samp_params['class_name']))))

        if self.term_weight=='TFIDF':
            doc_num_train = len(doc_class_list_train)
            df_term_train = pytc.stat_df_term(term_set_train,doc_terms_list_train)
            idf_term_train = pytc.stat_idf_term(doc_num_train,df_term_train)
        else:
            idf_term_train = []

        train_embeddings = []

        print "building training samps......"
        samp_list_train, class_list_train = pytc.build_samps(term_dict, class_dict, doc_class_list_train,
        doc_terms_list_train, doc_uni_token_train, self.term_weight, self.rule_feature, idf_term_train, train_embeddings)
        print "saving training samps......"
        pytc.save_samps(samp_list_train, class_list_train, fname_samps_train)


    def gen_test_samps(self, test_samp_params):
        fname_term_set = test_samp_params['term_set_dir'] + os.sep + 'term.set'
        fname_samps_test = test_samp_params['samp_dir'] +os.sep+'test.samp'
        # if not os.path.isfile(fname_term_set):
        #     print "cant find term set file."
        #     return
        if not os.path.exists(test_samp_params['raw_dir']):
            print "test dir does not exist."
            os.mkdir(test_samp_params['raw_dir'])

        doc_class_list_test, doc_str_token_test, doc_terms_list_test, term_set_test = \
        self.gen_doc_terms_list(test_samp_params, train_opt = 0)

        # unigram的token，用来构建其他规则特征
        doc_uni_token_test = pytc.gen_N_gram(doc_str_token_test,'uni')

        term_set_train = pytc.load_term_set(fname_term_set)
        term_dict = dict(zip(term_set_train, range(1,len(term_set_train)+1)))
        class_dict = dict(zip(test_samp_params['class_name'], range(1,1+len(test_samp_params['class_name']))))
        class_dict['test'] = 0

        if self.term_weight=='TFIDF':
            doc_num_test = len(doc_class_list_test)
            df_term_test = pytc.stat_df_term(term_set_test,doc_terms_list_test)
            idf_term_test = pytc.stat_idf_term(doc_num_test,df_term_test)
        else:
            idf_term_test = []

        test_embeddings = []
        print "building testing samps......"
        samp_list_test, class_list_test = pytc.build_samps(term_dict, class_dict, doc_class_list_test,
        doc_terms_list_test, doc_uni_token_test, self.term_weight, self.rule_feature, idf_term_test, test_embeddings)
        print "saving testing samps......"
        pytc.save_samps(samp_list_test, class_list_test, fname_samps_test)

    def N_folds_samps(self, input_dir, fold_num, train_params, test_params):
        '''将语料按照交叉验证的折数进行分割'''
        output_dir = input_dir+'_nfolds'
        pytc.gen_nfolds_f2(input_dir, output_dir, fold_num, train_params['token'])
        for fold_id in range(1, fold_num+1):
            print '\n\n##### Cross Validation: fold' + str(fold_id) + ' #####'
            fold_dir = output_dir + os.sep + 'fold' + str(fold_id)
            fold_train_dir = fold_dir + os.sep + 'train'
            fold_test_dir = fold_dir + os.sep + 'test'

            for w in ['raw_dir', 'samp_dir', 'term_set_dir', 'model_file_dir']:
                train_params[w] = fold_train_dir
            for w in ['term_set_dir', 'model_file_dir']:
                test_params[w] = fold_train_dir
            for w in ['raw_dir', 'samp_dir', 'result_file_dir']:
                test_params[w] = fold_test_dir
            self.gen_train_samps(train_params)
            self.gen_test_samps(test_params)

    def N_folds_validation(self, input_dir, fold_num, classifier_list,
    train_params, test_params):
        '''对每折验证中的语料进行训练与测试，并求融合模型的平均正确率'''
        output_dir = input_dir+'_nfolds'
        for fold_id in range(1,fold_num+1):
            fold_dir = output_dir + os.sep + 'fold' + str(fold_id)
            train_dir = fold_dir + os.sep + 'train'
            test_dir = fold_dir + os.sep + 'test'

            train_params['samp_dir'], train_params['model_file_dir'] = train_dir, train_dir
            test_params['samp_dir'], test_params['model_file_dir'], test_params['result_file_dir'] = test_dir, train_dir, test_dir
            for c in classifier_list:
                model_fname = c+'.model'
                result_fname = c+'.result'
                params.classifier_learn[c](train_params['samp_dir'], train_params['model_file_dir'], model_fname = model_fname)
                params.classifier_predict[c](test_params['samp_dir'], test_params['model_file_dir'],
            test_params['result_file_dir'], model_fname = model_fname, result_fname = result_fname)

if __name__ == '__main__':
    term_weight = 'BOOL'
    rule_feature = 0
    ngram_dict = {
        'token_gram':{
            'uni': {
                'df': 1,
                'fs_opt': 0,
                'fs_method': 'IG',
                'fs_num': 45000,
            },
            # 'bis':{
            #     'df': 2,
            #     'fs_opt': 0,
            #     'fs_method': 'IG',
            #     'fs_num': 10000,
            # },
            # 'tri':{
            #     'df': 1,
            #     'fs_opt': 1,
            #     'fs_method': 'IG',
            #     'fs_num': 10000,
            # },
        },
        'pos_gram':{
            # 'uni': {
            #     'df': 1,
            #     'fs_opt': 0,
            #     'fs_method': 'IG',
            #     'fs_num': 45000,
            # },
            # 'bis':{
            #     'df': 1,
            #     'fs_opt': 1,
            #     'fs_method': 'IG',
            #     'fs_num': 10000,
            # },
        },
        'tag_gram' : {
        },
        'character_gram': {

        }
    }


    train_params = {
        'token' : ['neg_fenci', 'pos_fenci'],
        'pos' : [],
        'tag' : [],
        'class_name' : ['neg', 'pos'],
        # 'train_raw_dir': train_raw_dir,
        # 'train_samp_dir': train_samp_dir,
        # 'term_set_dir': term_set_dir,
        # 'model_file_dir': model_file_dir,
    }

    test_params = {
        'token' : ['test_fenci'],
        'pos' : ['test_pos'],
        'tag' : ['test_tag'],
        'class_name' : ['test'],
        # 'test_samp_dir': '',
        # 'term_set_dir': term_set_dir,
        # 'model_file_dir': train_output_dir,
    }

    classifier_lst = ['lg']
    class_dict = {'1':'neg','2':'pos'}

    test = Mbsa(ngram_dict, term_weight, rule_feature)

    '''train'''
    train_dir = 'coae2014' + os.sep + 'train'
    train_dir = 'dbp' + os.sep + 'train'
    train_params['raw_dir'] = train_dir
    train_params['samp_dir'] = train_dir
    train_params['term_set_dir'] = train_dir
    train_params['model_file_dir'] = train_dir
    test.gen_train_samps(train_params)

    for c in classifier_lst:
        print c, "learning..."
        model_fname = c + '.model'
        params.classifier_learn[c](train_params['samp_dir'], train_params['model_file_dir'], model_fname = model_fname)



    '''test'''
    # test_dir = 'coae2014' + os.sep + 'test'
    test_dir = 'dbp' + os.sep + 'test'
    test_params['raw_dir'] = test_dir
    test_params['samp_dir'] = test_dir
    test_params['result_file_dir'] = test_dir
    test_params['term_set_dir'] = train_dir
    test_params['model_file_dir'] = train_dir
    test.gen_test_samps(test_params)

    for index, c in enumerate(classifier_lst):
        print c, "predicting..."
        model_fname = c + '.model'
        result_fname = c + '.result'
        params.classifier_predict[c](test_params['samp_dir'], test_params['model_file_dir'],
            test_params['result_file_dir'], model_fname = model_fname, result_fname = result_fname)

        '''performance'''
        start = 0
        if c == 'lg' or c == 'svm':
            start += 1
        result = [x.strip().split()[0] for x in open(test_params['result_file_dir'] + os.sep + c + '.result').readlines()[start:]]
        label = [x.strip() for x in open(test_params['raw_dir'] + os.sep + 'test_label').readlines()]
        result_dict = performance.demo_performance(result,label,class_dict)
        ss = ''
        for key in ['p_neg','r_neg','p_pos','r_pos','macro_f1','acc']:
            ss += str(round(result_dict[key]*100,4))+'%\t'
        ss = ss.rstrip('\t')
        print ss
