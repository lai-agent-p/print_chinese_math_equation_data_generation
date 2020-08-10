import cv2
import numpy as np

from src.util.file_util import load_file_dict, load_im_dict, load_lexicon, load_json, load_folder_dict, load_deletion_im_dict
from src.util.misc_util import is_chinese
from src.util.image_util import crop_and_padding, blur_image

import pdb


def load_character_dict(config):
    '''
    load character dict from json file

    Args:
        config: obj
            a class with all meta information

    Returns:
        ret_dict: Dict[str:Dict[str:str]]
            a dict of dict for all characters, the first key is the writer
             and the second key is for the character to pick. if you want to
             pick handwritten character '提' written by writer '228', you need
             ret_dict['228']['提']
    '''
    folders_dict = load_folder_dict(config.IM_FOLDERS_PATH)
    ret_dict = {}

    for folder_name, folder_path in folders_dict.items():
        writer_name = folder_name
        label_im_dict = load_im_dict(folder_path)
        if not writer_name in ret_dict.items():
            ret_dict[writer_name] = {}
        for i, (label, im_path) in enumerate(label_im_dict.items()):
            ret_dict[writer_name][label] = im_path

    return ret_dict


def load_crohme_dict(config):
    '''
    load crohme dict from json file

    Args:
        config: obj
            a class with all meta information

    Returns: Dict[List[str]]
        crohme_dict['α'][0]

    '''
    folders_dict = load_folder_dict(config.CROHME_IM_FOLDERS_PATH)
    ret_dict = {}

    for folder_name, folder_path in folders_dict.items():
        symbol = folder_name
        ret_dict[symbol] = []
        im_dict = load_im_dict(folder_path)
        for im_name, im_path in im_dict.items():
            ret_dict[symbol].append(im_path)
    return ret_dict


def load_source_labels(config, valid_math_symbols):
    '''

    Args:
        config: obj
            a class with all meta information
    Returns:
        label_samples: Dict[str, List[str]]
            a dict of labels, label is on the label list form
            ex:
            {16339652_option_c_0-0_normal_ppi: 900 ^ { \\circ }}, ...
            notice that some of the samples will also be dropped on future steps
     '''

    h_token2index, h_index2token = load_lexicon(config.HAND_LEXICON_PATH)
    raw_labels = load_json(config.LABELS_PATH)  # raw labels are data for type ocr

    hand_symbols = set(h_index2token)
    label_samples = {}

    for sample_name, meta in raw_labels.items():
        # token_list = cvtstring2tokenlist(meta['label'], diff_length_symbols)
        token_list = meta['label'].split(' ')
        str_list = []
        # for token in token_list:
        #     # if not token in valid_math_symbols and not is_chinese(token):
        #     if token == '{' or token == '}' or token == '[' or token == ']':
        #         #get rid of samples with token not in hand ocr dict
        #         str_list = []
        #         break
        #     else:
        #         str_list.append(token)

        if len(str_list) != 0:
            label_samples[sample_name] = token_list

    return label_samples


def get_chinese_symbols(character_dict):
    '''
    given character dict, get a set for all possible Chinese

    Args:
        character_dict: Dict[str:Dict[str:str]]
            a dict of dict for all characters, the first key is the writer
             and the second key is for the character to pick. if you want to
             pick handwritten character '提' written by writer '228', you need
             ret_dict['228']['提']
    Returns:
        Chinese_set: set[str]
            the set that contains all distinct Chinese that appears
    '''
    chinese_set = set()
    for author, hand_im_paths in character_dict.items():
        for token, hand_im_path in hand_im_paths.items():
            if is_chinese(token):
                chinese_set.add(token)
    return chinese_set


def get_valid_math_symbols(config):
    '''
    get all valid math symbols, for now we drop all symbols that
    requires arguments

    Args:
        config: obj
            The class that contains all meta information

    Returns:
        math_symbol_set: set[str]
            set of math symbols that may be inferenced
    '''
    return list(config.label2unicode.keys())


def get_drawable_math_symbols(config):
    '''
    get all drawable symbols, this one presents tokens in unicode format:
    ex the set doesn't contain \\times but it contains symbol ×
    Args:
        config: obj
            a class that contains all data

    Returns: set[str]
        a set that contains all strings
    '''

    numbers = list('0123456789')
    alphabets = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return set(list(config.unicodes_convert.keys()) + numbers + alphabets)


def load_deletion_symbols(config):
    deletion_symbols = load_im_dict(config.DELETION_SYMBOL_PATH)
    delete_im_dicts = {'short': [],
                       'alphabet': [],
                       'mid': [],
                       'long': []}

    for symbol_name, symbol_path in deletion_symbols.items():
        if 'alphabet' in symbol_name:
            im = cv2.imread(symbol_path, 0)
            im = 255 - crop_and_padding(255 - im)
            im = (255 - (255 - im) * 0.8).astype(np.uint8)
            im = blur_image(im)
            delete_im_dicts['alphabet'].append(im)
        elif 'short' in symbol_name:
            im = cv2.imread(symbol_path, 0)
            im = 255 - crop_and_padding(255 - im)
            im = (255 - (255 - im) * 0.8).astype(np.uint8)
            im = blur_image(im)
            delete_im_dicts['short'].append(im)
        elif 'mid' in symbol_name:
            im = cv2.imread(symbol_path, 0)
            im = 255 - crop_and_padding(255 - im)
            im = (255 - (255 - im) * 0.8).astype(np.uint8)
            im = blur_image(im)
            delete_im_dicts['mid'].append(im)
        elif 'long' in symbol_name:
            im = cv2.imread(symbol_path, 0)
            im = 255 - crop_and_padding(255 - im)
            im = (255 - (255 - im) * 0.8).astype(np.uint8)
            im = blur_image(im)
            delete_im_dicts['long'].append(im)
    return delete_im_dicts


def load_new_deletion_symbol(config):
    deletion_symbols = load_deletion_im_dict(config.NEW_DELETION_SYMBOL_PATH)
    delete_im_dicts = {'short': {'frac': [], 'sqrt': [], 'short': [], 'scription': []},
                       'one_char': {'frac': [], 'sqrt': [], 'one_char': [], 'scription': []},
                       'mid': {'frac': [], 'sqrt': [], 'mid': []},
                       'long': [],
                       'super_long': []}
    offset_dict = load_json(config.DELETION_OFFSET_PATH)
    for symbol_name, symbol_path in deletion_symbols.items():
        im = cv2.imread(symbol_path, 0)
        im = 255 - crop_and_padding(255 - im)
        im = (255 - (255 - im) * 0.8).astype(np.uint8)
        im = blur_image(im)
        temp = {'img': im, 'offset': offset_dict[symbol_path.split('/')[-1].split('.')[0]]}
        if 'one_char' in symbol_name:
            if 'rect' in symbol_name:
                delete_im_dicts['one_char']['frac'].append(temp)
            elif 'sqrt' in symbol_name:
                delete_im_dicts['one_char']['sqrt'].append(temp)
            elif 'scription' in symbol_name:
                delete_im_dicts['one_char']['scription'].append(temp)
            else:
                delete_im_dicts['one_char']['one_char'].append(temp)

        elif 'short' in symbol_name:
            if 'rect' in symbol_name:
                delete_im_dicts['short']['frac'].append(temp)
            elif 'sqrt' in symbol_name:
                delete_im_dicts['short']['sqrt'].append(temp)
            elif 'scription' in symbol_name:
                delete_im_dicts['short']['scription'].append(temp)
            else:
                delete_im_dicts['short']['short'].append(temp)

        elif 'mid' in symbol_name:
            if 'rect' in symbol_name:
                delete_im_dicts['mid']['frac'].append(temp)
            elif 'sqrt' in symbol_name:
                delete_im_dicts['mid']['sqrt'].append(temp)
            else:
                delete_im_dicts['mid']['mid'].append(temp)

        elif 'long' in symbol_name:
            delete_im_dicts['long'].append(temp)

        elif 'super_long' in symbol_name:
            delete_im_dicts['super_long'].append(im)
    return delete_im_dicts