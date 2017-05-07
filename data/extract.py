#!/usr/bin/env cpp
# -*- coding: utf-8 -*-
# @Date    : 2017-01-26 10:38:59
# @Author  : JieJ (jiej1992@163.com)

import sys,os

input_dir = sys.argv[1] + '_THULAC'
fenci = open(input_dir + os.sep + 'test_fenci').readlines()
label = [x.strip() for x in open(input_dir + os.sep + 'test_label').readlines()]

with open(input_dir + os.sep + 'neg_fenci', 'w') as ns, \
open(input_dir + os.sep + 'pos_fenci','w') as ps:
    for x, y in zip(fenci, label):
        if y == '1':
            ns.write(x)
        elif y == '2':
            ps.write(x)

