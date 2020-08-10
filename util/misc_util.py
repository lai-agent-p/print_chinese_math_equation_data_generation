import numpy as np
import os
from os.path import join
import cv2
import shutil
import time
import json


def save_json(save_obj, path, force_indent=True):
    """

    save an obj as json file to a path

    Args:

        save_obj: Any

            any object that can be converted to json format

        path: str

            out json file path

        force_indent: bool default: True

            set True to force using indent on the out json

    Returns:

        None

    """

    if force_indent:

        with open(path, 'w') as f:

            json.dump(save_obj, f, ensure_ascii=False, indent=4)

    else:

        with open(path, 'w') as f:

            json.dump(save_obj, f, ensure_ascii=False)


def first_zero(arr, axis, invalid_val=-1):
    '''
    find position of first zero on a array along a specified axis
     and return the positions as a list.

    :param arr: input is a n_d numpy array
    :param axis: along which axis we want to find first zero
    :param invalid_val: value for the case that no zero can be found on the axis
    :return: list of indices points to the position with first zero
    '''
    mask = arr == 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val).tolist()

class label_dict:
    def __init__(self, initial_dict):
        '''
        initialize the special dict
​
        :param initial_dict: The original token to
                             integer labels mapping.
        '''
        self.dict_ = initial_dict
        self.trash_val = initial_dict['<unk>']

    def __getitem__(self, key):
        '''
        operation function for [] for label dict class
        if the input token is in the lexicon, return the
        corresponding integer label, otherwise raise
        exceptiong or return <other> token depend on
        if the unknown character is Chinese
​
        :param key: the token to be mapped to an integer label.
        :return:
        '''
        if key in self.dict_.keys():
            return self.dict_[key]
        else:
            if key >= u'\u4e00' and key <= u'\u9fff':
                return self.trash_val
            else:
                print(key)
                raise ValueError('error on lexicon build')

    def __contains__(self, key):
        '''
        operation function to check if a key is in dictionary.
    ​
        :param key: test if the key is in dictionary
        :return: True if the key is in original lexicon or the key
                is a Chinese, otherwise return False
        '''
        if key in self.dict_.keys():
            return True
        else:
            if key >= u'\u4e00' and key <= u'\u9fff':
                return True
            else:
                return False


def run_nested_func(nested, func):
    '''
    run a function on each tensor on the nested struct

    :param nested: a nested struct, a list of element or a list of nested struct
    :param func: function to be applied on each element on the nested struct
    :return: result of the nested struct
    '''

    if not isinstance(nested, (list, tuple)):
        return func(nested)
    layer_out = []
    for sub_part in nested:
        if not isinstance(sub_part, (list, tuple)):
            sub_part = func(sub_part)
            layer_out.append(sub_part)
        else:
            sub_part = run_nested_func(sub_part, func)
            layer_out.append(sub_part)
    return layer_out


def editDistDP(str1, str2):
    '''
    edit distance between two sequences, where equivalent
    operation are well defined for the space of the elements
    aka for a \in str1, b \in str2, a==b is well defined

    :param str1: first string (or any kind of list)
    :param str2: second string (or any kind of list)
    :return: edit distance between the two lists
    '''
    m = len(str1)
    n = len(str2)
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    import pdb
    # pdb.set_trace()
    # Fill d[][] in bottom up manner
    for i in range(m + 1):
        for j in range(n + 1):

            # If first string is empty, only option is to
            # isnert all characters of second string
            if i == 0:
                dp[i][j] = j  # Min. operations = j

            # If second string is empty, only option is to
            # remove all characters of second string
            elif j == 0:
                dp[i][j] = i  # Min. operations = i

            # If last characters are same, ignore last char
            # and recur for remaining string
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]

            # If last character are different, consider all
            # possibilities and find minimum
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]


def retrieve_im(torch_im, mask):
    '''
    retrieve image

    :param torch_im: image in tensor form
    :param mask: mask in tensor form
    :return: image in numpy form
    '''
    im = torch_im.detach().cpu().numpy()[0] * 255
    mask = mask.detach().cpu().numpy()
    real_h = int(np.sum(mask, axis = 0)[0])
    real_w = int(np.sum(mask, axis = 1)[0])
    im = im[:real_h, :real_w]
    return im


def att_visualize(im, time_alphas, test_im_out_path, pre_name, pred_list, ind):
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

    h, w = im.shape
    im_path = join(test_im_out_path, str(ind))
    check_out_dir(im_path)
    for i, alpha in enumerate(time_alphas):
        att_map = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_NEAREST)
        normalized_map = cv2.normalize(att_map, np.zeros_like(att_map), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        im = im.astype(np.uint8)
        result = cv2.addWeighted(im, 0.3, normalized_map, 0.7, 0)
        cv2.imwrite(join(im_path, '{}_{}.png'.format(i, str(pred_list[i]).replace('</s>', 'end'))), result)

def get_time_str():
    '''
    get current time in a string format

    :return: a string indicates the current time
    '''
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime("%Y-%m-%d_%H:%M:%S", named_tuple)
    return time_string