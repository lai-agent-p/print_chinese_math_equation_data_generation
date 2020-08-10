'''
input and output utilities, this includes
all operations that involves communication
between disk and the program.
'''
import json
import os
import shutil
from src.util.misc_util import label_dict
# import diff_match_patch as dmp_module

def load_im_dict(image_folder):
    '''
    load all image paths to a folder, this function
    recursively traverse a folder and recursively get all
    image paths
    :param image_folder: image folder that may contain folders
    that contains image(currently supports png jpg and bmp)
    :return: a dict whose key is raw image name(as key for json files)
     and the dict's value is image path
    '''
    ret_dict = {}
    for root, dirs, files in os.walk(image_folder, topdown=False):
        for name in files:
            if name.lower().endswith(('png', 'jpg', 'bmp', 'jpeg')):
                raw_name = '.'.join(name.split('.')[:-1])
                ret_dict[raw_name] = os.path.join(root, name)

    return ret_dict

def check_out_dir(folder_path):
    '''
    make a dir if the input path doesn't exist, else
    empty the directory
    :param folder_path: path to the folder to be checked
    :return: None
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

def check_make_dir(folder_path):
    '''
    make a dir if the input path doesn't exist, else
    empty the directory
    :param folder_path: path to the folder to be checked
    :return: None
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def load_json(json_path):
    '''
    load one json file
    :param json_path: path to the json file, based on
    variable name the json file is assumed to be a
    dictionary.
    :return: load dictionary
    '''
    with open(json_path, 'r') as f:
        ret_dict = json.load(f)
    return ret_dict

def load_lexicon(lexicon_path):
    '''
    given a lexicon path, load lexicon and
    return a token2index map and a index2token map
    :param lexicon_path: path to the lexicon file, which is a list of tokens
    :return:token2index map and index2token map
    '''
    index2token = load_json(lexicon_path)
    # print(index2token)
    token2index = {}
    for i, token in enumerate(index2token):
        token2index[token] = i
    token2index = label_dict(token2index)
    return token2index, index2token

# def out_diff_html(file1, file2, html_path):
#     '''
#     get difference between file1 and file2 and output it
#     as a html
#     :param file1: file 1
#     :param file2: file 2
#     :return: None
#     '''
#     file1 = file1.encode().decode('UTF-8')
#     with open('gt.txt', 'w') as f:
#         f.write(file1)
#
#     file2 = file2.encode().decode('UTF-8')
#     with open('pred.txt', 'w') as f:
#         f.write(file2)
#
#     dmp = dmp_module.diff_match_patch()
#     dmp.Diff_Timeout = 0
#     diff = dmp.diff_main(file1, file2)
#     dmp.diff_cleanupSemantic(diff)
#     html = dmp.diff_prettyHtml(diff)
#     with open(html_path, 'w') as f:
#         f.write(html)

def load_deletion_list(path):
    '''
    load the list of image names that
    should be dropped
    :param path: path to the deletion list
    :return: a set that contains name for files that should be deleted.
    '''
    with open(path, 'r') as f:
        deletion_list = f.readlines()
    delete_set = set()
    for delete_file in deletion_list:
        delete_file = delete_file.strip()
        delete_set.add(delete_file)
    return delete_set



