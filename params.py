import os,sys
import pytc

term_weight = 'BOOL'
rule_feature = 0

ngram_dict = {
    'token_gram':{
        'uni': {
            'df': 1,
            'fs_opt': 0,
            'fs_method': 'MI',
            'fs_num': 50000,
        },
        # 'bis':{
        #     'df': 1,
        #     'fs_opt': 0,
        #     'fs_method': 'MI',
        #     'fs_num': 50000,
        # },
        # 'tri':{
        #     'df': 2,
        #     'fs_opt': 0,
        #     'fs_method': 'MI',
        #     'fs_num': 50000,
        # },

    },
    'pos_gram':{
    },
    'tag_gram' : {
    },
    'character_gram' : {
        # 'tri': {
        # },
    }
}

train_params = {
    'token' : ['neg_fenci','pos_fenci'],
    'pos' : [],
    'tag' : [],
    'class_name' : ['neg', 'pos'],
    # 'train_raw_dir': train_raw_dir,
    # 'train_samp_dir': train_samp_dir,
    # 'term_set_dir': term_set_dir,
    # 'model_file_dir': model_file_dir,
}

# train_params = {
#     'token' : ['neg_fenci','neu_fenci', 'pos_fenci'],
#     'pos' : [],
#     'tag' : [],
#     'class_name' : ['neg', 'neu', 'pos'],
#     # 'train_raw_dir': train_raw_dir,
#     # 'train_samp_dir': train_samp_dir,
#     # 'term_set_dir': term_set_dir,
#     # 'model_file_dir': model_file_dir,
# }


test_params = {
    'token' : ['test_fenci'],
    # 'token' : ['train_fenci'],
    'pos' : ['test_pos'],
    'tag' : ['test_tag'],
    'class_name' : ['test'],
    # 'test_samp_dir': '',
    # 'term_set_dir': term_set_dir,
    # 'model_file_dir': train_output_dir,
}

classifier_learn = {'lg':pytc.liblinear_learn, 'maxent':pytc.maxent_learn, 'nb':pytc.nb_learn, 'svm':pytc.libsvm_learn}
classifier_predict = {'lg':pytc.liblinear_predict, 'maxent':pytc.maxent_predict, 'nb':pytc.nb_predict, 'svm':pytc.libsvm_predict}


