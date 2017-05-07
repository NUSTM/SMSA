# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:02:50 2015

@author: JieJ
"""
import re,os

def transfer_emotions(fenci_fname):
    '''处理被切分开的表情词'''
    f = open(fenci_fname)
    text = f.readlines()
    f.close()

    f2 = open(fenci_fname,'w')
    for i in range(len(text)):
        lst1 = re.findall('(\[.*?\])',text[i]) # + re.findall('(#.*?#)',text[i])
        lst1 = list(set(lst1))
        new_text = text[i]
        if len(lst1)>=1:
            for w in lst1:
                x = w.replace(' ','')               #将空格替换掉
                new_text = new_text.replace(w,x)
        f2.write(new_text)
    f2.close()

if __name__ == '__main__':
    fname_list = []
    input_dir = 'Data\\SIGHAN2015_THULAC'
    fname_list.append(input_dir + os.sep + 'test_fenci')
    # fname_list.append(input_dir + os.sep + 'against_raw_fenci')
    # fname_list.append(input_dir + os.sep + 'none_raw_fenci')

    for fenci_fname in fname_list:
        transfer_emotions(fenci_fname)
    print 'over'
