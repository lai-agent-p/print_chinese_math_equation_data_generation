import json
import os
import shutil
import diff_match_patch as dmp_module
from os.path import isdir, join
from os import listdir

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

def save_json(json_path, save_obj):
    with open(json_path, 'w') as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=4)

def load_folder_dict(root_path):
    '''
    load all directories in a folder(non recursive)

    :param root_path: the folder that contains the directories
    :return: a dict whose keys are directory name and whose keys are directory paths
    '''
    dir_names = [dir_name for dir_name in listdir(root_path) if isdir(join(root_path, dir_name))]
    return {dir_name: join(root_path, dir_name) for dir_name in dir_names}

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

def load_deletion_im_dict(image_folder):
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
                raw_name = '_'.join(name.split('.')[:-1])
                ret_dict[raw_name] = os.path.join(root, name)
    return ret_dict

def load_file_dict(image_folder):
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
            if name.lower().endswith(('png', 'jpg', 'bmp', 'jpeg', 'txt')):
                ret_dict[name] = os.path.join(root, name)
    return ret_dict

def load_lexicon(lexicon_path):
    '''
    given a lexicon path, load lexicon and
    return a token2index map and a index2token map
    :param lexicon_path: path to the lexicon file, which is a list of tokens
    :return:token2index map and index2token map
    '''
    index2token = load_json(lexicon_path)
    token2index = {}
    for i, token in enumerate(index2token):
        token2index[token] = i
    return token2index, index2token
