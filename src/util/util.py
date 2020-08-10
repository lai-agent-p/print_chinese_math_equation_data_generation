import unicodedata
import numpy as np
import cv2
import sys
import pdb


def unicode_to_symbol(char):
    if char == '20D7':
        char = '2192'
    elif char == '2288':
        char = '2282'
    if char == '2218':
        symbol = 'o'
    elif char == '2212':
        symbol = '-'
    elif char == '5F':
        symbol = '\\_'
    elif char == 'AF':
        symbol = '-'
    elif char == '2E' or char == '22C5':
        symbol = ','
    elif char == '2223':
        symbol = '1'
    elif char == '2282':
        symbol = 'c'
    elif char =='22EF':
        symbol = '…'
    else:
        symbol = unicodedata.normalize("NFKD", chr(int(char, 16)))
    return symbol


def preprocess_labels(label, config):
    label = label.replace('\(', '(')
    label = label.replace('\)', ')')
    label = label.replace('\]', ']')
    label = label.replace('\[', '[')
    label = label.replace('\\\\', '')
    label = label.replace('\\ln', 'l n')
    label = label.replace('\\cos', 'c o s')
    label = label.replace('\\sin', 's i n')
    label = label.replace('\\cot', 'c o t')
    label = label.replace('\\tan', 't a n')
    label = label.replace('\\cosec', 'c o s e c')
    label = label.replace('\\sec', 's e c')
    label = label.replace('\\min', 'm i n')
    label = label.replace('\\max', 'm a x')
    label = label.replace('\\log', 'l o g')
    label = label.replace('\\nsubset', '\\nsubseteq')
    if label.split()[0] == '}':
        label = ' '.join(label.split()[1:])
    if '{' in label:
        # pdb.set_trace()
        label_list = label.split()
        try:
            brac_pairs = find_brac_pairs(label_list)
        except:
            return label
        frac_den_brac = []
        for key in brac_pairs.keys():
            if frac_den_brac != [] and key in frac_den_brac:
                frac_den_brac.remove(key)
                continue
            if key != 0:
                if label_list[key - 1] not in config.ONE_ARG_SYMBOLS and label_list[key - 1] not in config.TWO_ARG_SYMBOLS and label_list[key - 1] != ']':
                    label_list[key] = '\\{'
                    label_list[brac_pairs[key]] = '\\}'
                if label_list[key - 1] == '\\frac':
                    # pdb.set_trace()
                    frac_den_brac.append(brac_pairs[key] + 1)
            else:
                label_list[key] = '\\{'
                label_list[brac_pairs[key]] = '\\}'
        label = ' '.join(label_list)
    if '^' in label and '_' in label:
        label = convert_scription(label.split())
    return label


def postprocess_labels(label):
    label = label.replace('(', '\\(')
    label = label.replace(')', '\\)')
    label = label.replace(']', '\\]')
    label = label.replace('[', '\\[')
    label = label.replace('l n', '\\ln')
    label = label.replace('c o s', '\\cos')
    label = label.replace('s i n', '\\sin')
    label = label.replace('c o t', '\\cot')
    label = label.replace('t a n', '\\tan')
    label = label.replace('c o s e c', '\\cosec')
    label = label.replace('s e c', '\\sec')
    label = label.replace('m i n', '\\min')
    label = label.replace('m a x', '\\max')
    label = label.replace('l o g', '\\log')
    label = label.replace('\\nsubseteq', '\\nsubset')
    return label


def convert_scription(label):
    '''
    covert labels for subscription following supersciption to supersciption following subscription 
    e.g. C _ { n } ^ { m - r } ) = C _ { 2 n } ^ { m } . to 
         C ^ { m - r } _ { n } ) = C ^ { m } _ { 2 n } .
    '''
    try:
        brac_pairs = find_brac_pairs(label)
        insert = []
        indices = []
        del_ind = []
        new_dict = {}
        del_len = 0
        for key, val in brac_pairs.items():
            if label[key - 1] == '_':
                if label[val + 1] == '^':
                    insert.append(label[key - 1:val + 1])
                    new_dict[key - del_len] = val - del_len
                    del_ind.append(key - del_len)
                    del_len = len(label[key - 1:val + 1])
                    indices.append(brac_pairs[val + 2] - del_len + 1)
        for i in del_ind:
            del label[i - 1:new_dict[i] + 1]
        for i, ele in enumerate(insert):
            ind = indices[i]
            for j in ele:
                label.insert(ind, j)
                ind += 1
        return ' '.join(label)
    except:
        return ' '.join(label)


def find_brac_pairs(label):
    '''
    Find the pairs of open and close brackets
    '''
    istart = []  # stack of indices of opening parentheses
    d = {}
    for i, c in enumerate(label):
        if c == '{':
            istart.append(i)
        if c == '}':
            d[istart.pop()] = i
    return d


def cal_leaf_nodes(node, num_nodes):
    '''
    calculates the number of leaf nodes in a sub tree
    '''
    for sub_node in node:
        try:
            sub_node = sub_node[0]
        except:
            pass
        try:
            sub_node['type']
        except:
            pdb.set_trace()
        if sub_node['type'] == 'leaf':
            num_nodes += 1
        elif sub_node['type'] == 'frac':
            num_nodes = cal_leaf_nodes(sub_node['num'][0], num_nodes)
            num_nodes = cal_leaf_nodes(sub_node['den'][0], num_nodes)
        elif sub_node['type'] == 'sqrt':
            num_nodes = cal_leaf_nodes(sub_node['content'][0], num_nodes)
        else:
            num_nodes = cal_leaf_nodes(sub_node['content'][0], num_nodes)
    return num_nodes
