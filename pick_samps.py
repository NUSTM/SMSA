# -*- coding: utf-8 -*-
import os
import random
import sys


def random_sample(input_dir, output_dir, src_fname, tar_fname, samp_num):
    lst = open(input_dir+os.sep+src_fname).readlines()
    index_lst = range(len(lst))
    random.shuffle(index_lst)
    pick_lines = [lst[i] for i in index_lst[:samp_num]]
    with open(output_dir+os.sep+tar_fname, 'w') as xs:
        xs.writelines(pick_lines)




def random_sample_lst(lst, output_dir, tar_fname, samp_num, start_index=-1):
    index_lst = range(len(lst))
    random.shuffle(index_lst)
    if start_index == -1:
        pick_lines = [lst[i] for i in index_lst[:samp_num]]
    else:
        pick_lines = [lst[i] for i in index_lst[start_index:start_index+samp_num]]
    with open(output_dir+os.sep+tar_fname, 'w') as xs:
        xs.writelines(pick_lines)

def random_sample_lst_2(lst, index_lst, output_dir, tar_fname, samp_num, start_index=-1):
    if start_index >= len(index_lst):
        print "out of length!"
        return []
    elif start_index == 0:
        pick_lines = [lst[i] for i in index_lst[:samp_num]]
    else:
        pick_lines = [lst[i] for i in index_lst[start_index:start_index+samp_num]]
    # print len(pick_lines)
    # return pick_lines
    with open(output_dir+os.sep+tar_fname, 'w') as xs:
        xs.writelines(pick_lines)

def random_sample_lst_3(lst, index_lst, output_dir, tar_fname, samp_num, start_index=-1):
    random.shuffle(index_lst)
    if start_index >= len(index_lst):
        print "out of length!"
        return []
    elif start_index == 0:
        pick_lines = [lst[i] for i in index_lst[:samp_num]]
    else:
        pick_lines = [lst[i] for i in index_lst[start_index:start_index+samp_num]]
    # print len(pick_lines)
    # return pick_lines
    with open(output_dir+os.sep+tar_fname, 'w') as xs:
        xs.writelines(pick_lines)

def random_sample_with_weight(input_dir, output_dir, src_fenci_fname,
    tar_fenci_fname, src_weight_fname, tar_weight_fname, samp_num):
    fenci_lst = open(input_dir + os.sep + src_fenci_fname).readlines()
    weight_lst = open(input_dir + os.sep + src_weight_fname).readlines()
    if len(fenci_lst) != len(weight_lst):
        print "samps' quantity doesnt match samp weight!"
        return -1

    index_lst = range(len(fenci_lst))
    random.shuffle(index_lst)
    pick_fenci_lines = [fenci_lst[i] for i in index_lst[:samp_num]]
    pick_weight_lines = [weight_lst[i] for i in index_lst[:samp_num]]

    with open(output_dir + os.sep + tar_fenci_fname, 'w') as xs,\
    open(output_dir + os.sep + tar_weight_fname, 'w') as ws:
        xs.writelines(pick_fenci_lines)
        ws.writelines(pick_weight_lines)
    return 1;

def similar_sample(input_dir, output_dir, src_fname, tar_fname, samp_num):
    lst = open(input_dir+os.sep+src_fname).readlines()
    f = open(output_dir+os.sep+tar_fname, 'w')
    f.writelines(lst[:samp_num])
    f.close()


def index_sample(input_dir, output_dir, src_fname, tar_fname, index_lst):
    lst = open(input_dir + os.sep + src_fname).readlines()
    f = open(output_dir + os.sep + tar_fname, 'w')
    for index in index_lst:
        f.write(lst[index])
    f.close()
