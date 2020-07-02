#!/usr/bin/env python

import os
import sys
import pickle as pkl
import numpy


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0],gtd_list[i][1],gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right','Above','Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub','Below']:
                    return_string += ['_','{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup','Above']:
                    return_string += ['^','{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string

latex_root_path = '***'
gtd_root_path = '***'
gtd_paths = ['test_caption_14','test_caption_16','test_caption_19']

for gtd_path in gtd_paths:
    gtd_files = os.listdir(gtd_root_path + gtd_path + '/')
    f_out = open(latex_root_path + gtd_path + '.txt', 'w')
    for process_num, gtd_file in enumerate(gtd_files):
        # gtd_file = '510_em_101.gtd'
        key = gtd_file[:-4] # remove .gtd
        f_out.write(key + '\t')
        gtd_list = []
        gtd_list.append(['<s>',0,-1,'root'])
        with open(gtd_root_path + gtd_path + '/' + gtd_file) as f:
            lines = f.readlines()
            for line in lines[:-1]:
                parts = line.split()
                sym = parts[0]
                childid = int(parts[1])
                parentid = int(parts[3])
                relation = parts[4]
                gtd_list.append([sym,childid,parentid,relation])
        latex_list = convert(1, gtd_list)
        if 'illegal' in latex_list:
            print (key + ' has error')
            latex_string = ' '
        else:
            latex_string = ' '.join(latex_list)
        f_out.write(latex_string + '\n')
        # sys.exit()

    if (process_num+1) // 2000 == (process_num+1) * 1.0 / 2000: 
        print ('process files', process_num)
