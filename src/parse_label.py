import numpy as np
import cv2

from src.util.define_nodes import nodes
from src.util.util import find_brac_pairs
from project_config import ProjectConfig
import pdb

node = nodes()
remove_elements = ['{', '}', '^', '_', '[', ']']
            
    
def process_leaf(char, lab):
    # print(char)
    element = node.leaf_node()
    element['content'] = char['id']
    element['offset'] = char['offset']
    if 'scale' in char:
        element['scale'] = char['scale']
    else:
        element['scale'] = 1
    element['h'] = char['h']
    element['w'] = char['w']
    element['label'] = lab
    return element
        
       
def parse_label(label_list, virtual_image, tree):
    '''
    Parsing label to tree like structure and finding corresponding virtual image

    Args:
        label_list: List[str]
            list of label elements
        virtual_image: List[Dict]
            List of all character information's extracted from svg
        tree: List[Dict.....]
            Tree like structure that has nodes and subnodes
            eg. label: 5 + \\sqrt { 4 }
            tree = [{'type': 'leaf', 'label'='5',...},
                    {'type': 'sqrt', label='\\sqrt'..., 'content': ['type':leaf, label=4, ...]}]
            please see src/util/define_nodes for all possible nodes
    Returns: tree
    '''

    brac_pairs = find_brac_pairs(label_list)
    # print(brac_pairs)
    img_pos = 0
    label_pos = 0
    # pdb.set_trace()
    while label_pos < len(label_list):
        item = label_list[label_pos]
        # print(item)
        # print(virtual_image[img_pos])
        if item == '\\frac':
            # pdb.set_trace()
            label_pos+=1
            element = node.frac_node()
            label_items = label_list[label_pos+1:brac_pairs[label_pos]]
            vim_count = len([i for i in label_items if i not in remove_elements])
            if '\\sqrt' in label_items:
                sqrt_count = label_items.count('\\sqrt')
                vim_count+=sqrt_count
                vim_list = virtual_image[img_pos+1:img_pos+1+vim_count]                
            else:
                vim_list = virtual_image[img_pos+1:img_pos+1+vim_count]
            element['num_offset'] = virtual_image[img_pos]['num_offset']
            element['den_offset'] = virtual_image[img_pos]['den_offset']
            element['global_offset'] = virtual_image[img_pos]['global_offset']
            element['line_sym'] = virtual_image[img_pos]['id']
            element['num'].append(parse_label(label_items, vim_list, []))
            label_pos = brac_pairs[label_pos] + 1
            img_pos += vim_count+1
            label_items = label_list[label_pos+1:brac_pairs[label_pos]]
            vim_count = len([i for i in label_items if i not in remove_elements])
            if '\\sqrt' in label_items:
                sqrt_count = label_items.count('\\sqrt')
                vim_count+=sqrt_count
                vim_list = virtual_image[img_pos:img_pos+vim_count]
            else:
                vim_list = virtual_image[img_pos:img_pos+vim_count]
            # print(label_items)
            # print([i['id'] for i in vim_list[:-1]])           
            element['den'].append(parse_label(label_items, vim_list, []))
            label_pos = brac_pairs[label_pos] + 1
            img_pos += vim_count     
            
        elif item == '\\sqrt':
            # pdb.set_trace()
            label_pos += 1
            if label_list[label_pos] == '[':
                root_num_label = label_list[label_pos+1]
                label_pos+=3
            element = node.sqrt_node()
            label_items = label_list[label_pos+1:brac_pairs[label_pos]]
            vim_count = len([i for i in label_items if i not in remove_elements])
            if '\\sqrt' in label_items:
                sqrt_count = label_items.count('\\sqrt')
                vim_count+=sqrt_count
                vim_list = virtual_image[img_pos+1:img_pos+1+vim_count]
            else:
                # vim_count+=1
                vim_list = virtual_image[img_pos+1:img_pos+1+vim_count]
            # print(label_items)
            # print([i['id'] for i in vim_list[:-1]])
            element['line_sym'] = virtual_image[img_pos]['id']
            element['line_w'] = virtual_image[img_pos]['w']
            if virtual_image[img_pos+vim_count+1]['id'].split('-')[-1] == '221A':
                surd_ind = img_pos+vim_count+1
            else:
                element['root_num']['num'] = virtual_image[img_pos+vim_count+1]['id']
                element['root_num']['offset'] = virtual_image[img_pos+vim_count+1]['offset']
                element['root_num']['h'] = virtual_image[img_pos+vim_count+1]['h']
                element['root_num']['w'] = virtual_image[img_pos+vim_count+1]['w']                  
                element['root_num']['label'] = root_num_label                
                surd_ind = img_pos+vim_count+2
                img_pos+=1
            element['sqrt_sym']['sym'] = virtual_image[surd_ind]['id']
            element['sqrt_sym']['offset'] = virtual_image[surd_ind]['offset']
            element['sqrt_sym']['h'] = virtual_image[surd_ind]['h']
            element['sqrt_sym']['w'] = virtual_image[surd_ind]['w']            
            element['content'].append(parse_label(label_items, vim_list, []))
            label_pos = brac_pairs[label_pos] + 1 
            img_pos += vim_count+2
                
        elif item == '^' or item == '_':
            label_pos+=1
            element = node.scription_node()
            element['label'] = item
            label_items = label_list[label_pos+1:brac_pairs[label_pos]]
            vim_count = len([i for i in label_items if i not in remove_elements])
            if '\\sqrt' in label_items:
                sqrt_count = label_items.count('\\sqrt')
                vim_count+=sqrt_count
                vim_list = virtual_image[img_pos:img_pos+vim_count]
            else:
                vim_list = virtual_image[img_pos:img_pos+vim_count]            
            # print(label_items)
            # print([i['id'] for i in vim_list[:-1]])
            element['content'].append(parse_label(label_items, vim_list, [])) 
            label_pos = brac_pairs[label_pos] + 1
            img_pos += vim_count
            
        elif item in ProjectConfig.ONE_ARG_SYMBOLS:
            # pdb.set_trace()
            label_pos+=1
            element = node.one_arg_node()
            label_items = label_list[label_pos+1:brac_pairs[label_pos]]
            vim_count = len([i for i in label_items if i not in remove_elements])
            if '\\sqrt' in label_items:
                sqrt_count = label_items.count('\\sqrt')
                vim_count+=sqrt_count
                vim_list = virtual_image[img_pos:img_pos+vim_count]
            else:
                vim_list = virtual_image[img_pos:img_pos+vim_count]
            element['special_sym']['sym'] = virtual_image[img_pos+vim_count]['id']
            element['special_sym']['offset'] = virtual_image[img_pos+vim_count]['offset']
            element['special_sym']['h'] = virtual_image[img_pos+vim_count]['h']
            element['special_sym']['w'] = virtual_image[img_pos+vim_count]['w']
            element['special_sym']['label'] = item
            element['left_offset'] = virtual_image[img_pos+vim_count]['left_offset']
            element['right_offset'] = virtual_image[img_pos+vim_count]['right_offset']
            element['global_offset'] = virtual_image[img_pos+vim_count]['global_offset']
            element['content'].append(parse_label(label_items, vim_list, [])) 
            label_pos = brac_pairs[label_pos] + 1 
            img_pos += vim_count+1
            
        else:
            element = process_leaf(virtual_image[img_pos], label_list[label_pos])
            label_pos+=1
            img_pos+=1
        tree.append(element)    
    return tree

