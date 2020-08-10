import numpy as np
import random
import os
import cv2
import math
from src.inference_im_model import inference_im_model
from src.inference_im_model import load_model
import pdb

from src.util.util import unicode_to_symbol
from src.util.image_util import fix_lr_boundary, deskew, erode_plus
from src.util.misc_util import is_chinese
from src.util.augmentation import random_shear_rotation, morphological_operations


def inference_char(node, characters_dict, crohme_path_dict, config, author, curr_pos, prev_h, prev_w,
                   offset_position, in_den):
    """
    inference the character other than chinese
    Args:
        node: Dict
            it contains all the informations to inference character
            for eg. h, w, ascii, label, offset
        characters_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        crohme_path_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        config: config file
        author: int
            selected author
        curr_pos: Tuple
            previous character end position
        prev_h: int
            height of previous character
        prev_w: int
            width of previous character
        offset_position: str
            up or middle
        in_den: bool
            current subnode is denominator
    Returns: img, lu position, size and next_pos
    """
    id = node['content'].split('-')[-1]
    try:
        encoded_char = unicode_to_symbol(id)
    except:
        encoded_char = id
        # raise Exception('unicode conversion error')
    if encoded_char in config.crohme_keys:
        if encoded_char == '/':
            encoded_char = 'forward_slash'
        im_path = random.choice(crohme_path_dict[encoded_char])
    else:
        if not encoded_char in characters_dict[author].keys():
            # print('not in the directory ', id)
            # print(encoded_char)
            raise Exception('{} not in the directory'.format(encoded_char))
        im_path = characters_dict[author][encoded_char]
    char_box = cv2.imread(im_path, 0)
    h, w = char_box.shape
    if config.ONE_CHARACTER_SIZE_MODE == 'svg':
        if w / node['w'] > 0.7:
            w = node['w']
        if h / node['h'] > 0.7:
            h = node['h']
    if id == '221A':
        h = node['h']
        w = node['w']
    # h = random.randint(math.ceil(h * 0.95), math.ceil(h * 1.05))
    # w = random.randint(math.ceil(w * 0.90), math.ceil(w * 1.1))
    if node['scale'] is not None:
        h = math.ceil(h * node['scale'])
        w = math.ceil(w * node['scale'])
    min_w = min(prev_w, w)
    min_h = min(prev_h, h)
    # shift_x = random.randint(-math.ceil(min_w * 0.1), math.ceil(min_w * 0.1))
    # shift_y = random.randint(-math.ceil(min_h * 0.01), math.ceil(min_h * 0.01))
    shift_y = 0
    char_box = cv2.resize(char_box, (w, h))
    if id == '2288':
        extra_im_path = random.choice(crohme_path_dict['forward_slash'])
        extra_im = cv2.imread(extra_im_path, 0)
        extra_im = cv2.resize(extra_im, (w, h))
        char_box = np.minimum(extra_im, char_box)
    offset = node['offset']
    # print(offset)
    if not in_den:
        shift_x = random.randint(-abs(offset[0]), 0)
    else:
        shift_x = random.randint(-abs(math.floor(offset[0] * 0.1)), abs(math.ceil(offset[0] * 0.1)))
        # shift_y = random.randint(-abs(math.floor(offset[1] * 0.5)), 0)
    if encoded_char == ',':
        shift_x += random.randint(5, 8)
    if offset_position == 'middle':
        lu_pos = (int(curr_pos[0] + offset[0] + shift_x), int(curr_pos[1] + offset[1] - h // 2 + shift_y))
        next_pos = (lu_pos[0] + w, lu_pos[1] + h // 2)
    elif offset_position == 'up':
        lu_pos = (int(curr_pos[0] + offset[0] + shift_x), int(curr_pos[1] + offset[1] + shift_y))
        next_pos = (lu_pos[0] + w, lu_pos[1])
    # char_box = random_shear_rotation(char_box)
    # char_box = morphological_operations(char_box)
    return char_box, lu_pos, [h, w], next_pos, True


def inference_text(node, characters_dict, config, author, curr_pos, prev_h, prev_w, offset_position):
    """
    inference the chinese character
    Args:
        node: Dict
            it contains all the informations to inference character
            for eg. h, w, ascii, label, offset
        characters_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        config: config file
        author: int
            selected author
        curr_pos: Tuple
            previous character end position
        prev_h: int
            height of previous character
        prev_w: int
            width of previous character
        offset_position: str
            up or middle
    Returns: img, lu position, size and next_pos
    """
    token = node['content']
    # print(token)
    if not token in characters_dict[author].keys():
        # print('not in the directory', token)
        raise Exception('not in the directory ')
    im_path = characters_dict[author][token]
    text_box = cv2.imread(im_path, 0)
    h, w = text_box.shape
    if config.ONE_CHARACTER_SIZE_MODE == 'svg':
        if w / node['w'] > 0.7:
            w = node['w']
        if h / node['h'] > 0.7:
            h = node['h']
    h = random.randint(math.ceil(h * 0.80), math.ceil(h * 1.5))
    w = random.randint(math.ceil(w * 0.90), math.ceil(w * 1.1))
    if node['scale'] is not None:
        h = math.ceil(h * node['scale'])
        w = math.ceil(w * node['scale'])
    min_w = min(prev_w, w)
    min_h = min(prev_h, h)
    shift_x = random.randint(math.ceil(min_w * 0.2), math.ceil(min_w * 0.3))
    shift_y = random.randint(-math.ceil(min_h * 0.1), math.ceil(min_h * 0.1))
    text_box = cv2.resize(text_box, (w, h))
    offset = node['offset']
    # print(h,w)
    if offset_position == 'middle':
        # lu_pos = (curr_pos[0], curr_pos[1] - h//2)
        lu_pos = (int(curr_pos[0] + offset[0] + shift_x), int(curr_pos[1] + offset[1] - h // 2 + shift_y))
        next_pos = (lu_pos[0] + w, lu_pos[1] + h // 2)
    elif offset_position == 'up':
        lu_pos = (int(curr_pos[0] + offset[0] + shift_x), int(curr_pos[1] + offset[1] + shift_y))
        next_pos = (lu_pos[0] + w, lu_pos[1])
    # text_box = random_shear_rotation(text_box)
    # text_box = morphological_operations(text_box)
    return text_box, lu_pos, [h, w], next_pos, True


def inference_leaves(nodes, canvas, characters_dict, config, author, curr_pos, prev_h, prev_w, offset_position, model):
    if not offset_position == 'middle':
        raise ValueError('must set offset_position to middle if using layout model')
    ims = []
    sizes = []
    orig_lu_poses = []
    cur_label = []
    for node in nodes['children']:
        # print(node)
        token = node['content']
        if len(token.split('-')) > 1:
            id = token.split('-')[-1]
            try:
                token = unicode_to_symbol(id)
            except:
                token = id
        cur_label.append(token)
        # print(token)
        if not token in characters_dict[author].keys():
            # print('not in the directory', token)
            raise Exception('{} not in the directory'.format(token))
        im_path = characters_dict[author][token]
        text_box = cv2.imread(im_path, 0)
        h, w = text_box.shape
        if config.ONE_CHARACTER_SIZE_MODE == 'svg':
            if w / node['w'] > 0.7:
                w = node['w']
            if h / node['h'] > 0.7:
                h = node['h']
        # if id == '221A':
        #     h = node['h']
        #     w = node['w']
        h = random.randint(math.ceil(h * 0.90), math.ceil(h * 1.1))
        w = random.randint(math.ceil(w * 0.90), math.ceil(w * 1.1))
        min_w = min(prev_w, w)
        min_h = min(prev_h, h)
        shift_x = random.randint(math.ceil(min_w * 0.1), math.ceil(min_w * 0.2))
        shift_y = random.randint(-math.ceil(min_h * 0.1), math.ceil(min_h * 0.1))
        text_box = cv2.resize(text_box, (w, h))
        sizes.append((h, w))
        offset = node['offset']
        lu_pos = (int(curr_pos[0] + offset[0] + shift_x), int(curr_pos[1] + offset[1] - h // 2 + shift_y))
        orig_lu_poses.append(lu_pos)
        next_pos = (lu_pos[0] + w, lu_pos[1] + h // 2)
        ims.append(text_box)

    lu_poses, next_poses = inference_im_model(ims, cur_label, orig_lu_poses[0], model)
    return ims, lu_poses, sizes, next_poses, True


def inference_rect(rect, canvas, characters_dict, crohme_path_dict, config, author, box_w):
    """
    inference rect box for square root and fractions

    Args:
        rect: Dict
       canvas: Tuple
            canvas start and end points
        characters_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        crohme_path_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        config: config file
        author: int
            selected author
        box_w: int
        width of the box
    Returns: img, lu position, size and next_pos
    """
    token = '-'
    if not token in characters_dict[author].keys():
        # print('not in the directory', token)
        raise Exception('not in the directory ')
    im_path = characters_dict[author][token]
    rect_box = cv2.imread(im_path, 0)
    h, w = (5, box_w + 10)
    rect_box = cv2.resize(rect_box, (w, h))
    # rect_box = random_shear_rotation(rect_box)
    # rect_box = morphological_operations(rect_box)
    return rect_box, [h, w], True


def inference_deletion(node, characters_dict, config, author, curr_pos, prev_h, prev_w, offset_position,
                       delete_ims):
    """
    inference deleted images
    Args:
        node: Dict
            it contains all the informations to inference character
            for eg. h, w, ascii, label, offset
        canvas: Tuple
            canvas start and end points
        characters_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        crohme_path_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        config: config file
        author: int
            selected author
        curr_pos: Tuple
            previous character end position
        prev_h: int
            height of previous character
        prev_w: int
            width of previous character
        offset_position: str
            up or middle
        delete_ims: Dict[images]
            deletion symbols dictionary
    Returns: img, lu position, size and next_pos
    """
    del_chars = node['content']
    del_boxes = []
    # h = random.randint(math.ceil(node['h'] * 0.9), math.ceil(node['h'] * 1.1))
    # w = random.randint(math.ceil(node['w'] * 0.9), math.ceil(node['w'] * 1.1))
    for token in del_chars:
        if token not in characters_dict[author].keys():
            raise Exception('{} not in the directory'.format(token))
        im_path = characters_dict[author][token]
        sample_box = cv2.imread(im_path, 0)
        h, w = sample_box.shape
        # if w / node['w'] < 0.7:
        #     node['w'] = w
        # if h / node['h'] < 0.7:
        #     node['h'] = h
        sample_box = cv2.resize(sample_box, (node['w'], node['h']))
        # sample_box, thickness = erode_plus(sample_box)
        del_boxes.append(sample_box)
    del_boxes = np.concatenate(del_boxes, axis=1)
    if len(del_chars) == 1:
        delete_sample = random.choice(delete_ims['one_char']['one_char'])
        # delete_sample = delete_ims['one_char']['one_char'][5]
    elif len(del_chars) < 3:
        delete_sample = random.choice(delete_ims['short']['short'])
        # delete_sample = delete_ims['short']['short'][0]
        # pdb.set_trace()
    elif len(del_chars) < 5:
        delete_sample = random.choice(delete_ims['mid']['mid'])
    elif len(del_chars) < 8:
        delete_sample = random.choice(delete_ims['long'])
    else:
        delete_sample = random.choice(delete_ims['super_long'])
    # pdb.set_trace()
    del_offset = delete_sample['offset']
    ratio = del_offset['ratio']
    con_h, con_w = del_boxes.shape
    temp_ratio = [con_w / del_offset['w'], con_h / del_offset['h']]
    con_lu_pos = [0, 0]
    del_lu_pos = [-int(del_offset['left_up_w']*temp_ratio[0]), -int(del_offset['left_up_h']*temp_ratio[1])]
    # del_lu_pos = [0, 0]
    min_x = min(con_lu_pos[0], del_lu_pos[0])
    min_y = min(con_lu_pos[1], del_lu_pos[1])
    con_lu_pos = [int(con_lu_pos[0] + abs(min_x)), int(con_lu_pos[1] + abs(min_y))]
    del_lu_pos = [int((del_lu_pos[0] + abs(min_x))), int((del_lu_pos[1] + abs(min_y)))]
    con_rd_pos = [con_lu_pos[0] + con_w, con_lu_pos[1] + con_h]
    del_rd_pos = [con_rd_pos[0] - int(del_offset['right_down_w']*temp_ratio[0]), con_rd_pos[1] - int(del_offset['right_down_h']*temp_ratio[1])]
    delete_sample_img = cv2.resize(delete_sample['img'], (del_rd_pos[0] - del_lu_pos[0], del_rd_pos[1] - del_lu_pos[1]))
    del_h, del_w = delete_sample_img.shape
    sub_canvas_w = max(con_lu_pos[0] + con_w, del_lu_pos[0] + del_w)
    sub_canvas_h = max(con_lu_pos[1] + con_h, del_lu_pos[1] + del_h)
    sub_canvas = np.ones((sub_canvas_h, sub_canvas_w)) * 255
    sub_canvas[con_lu_pos[1]: con_lu_pos[1] + con_h, con_lu_pos[0]: con_lu_pos[0] + con_w] = del_boxes
    sub_canvas[del_lu_pos[1]: del_lu_pos[1] + del_h, del_lu_pos[0]: del_lu_pos[0] + del_w] = np.minimum(
        sub_canvas[del_lu_pos[1]: del_lu_pos[1] + del_h, del_lu_pos[0]: del_lu_pos[0] + del_w],
        delete_sample_img)
    offset = node['offset']
    min_w = min(prev_w, w)
    min_h = min(prev_h, h)
    shift_x = random.randint(math.ceil(min_w * 0.1), math.ceil(min_w * 0.3))
    shift_y = random.randint(-math.ceil(min_h * 0.1), math.ceil(min_h * 0.1))
    if offset_position == 'middle':
        lu_pos = (int(curr_pos[0] + offset[0]), int(curr_pos[1] + offset[1] - sub_canvas_h // 2))
        next_pos = (lu_pos[0] + sub_canvas_w, lu_pos[1] + sub_canvas_h // 2)
    elif offset_position == 'up':
        lu_pos = (int(curr_pos[0] + offset[0] ), int(curr_pos[1] + offset[1]))
        next_pos = (lu_pos[0] + sub_canvas_w, lu_pos[1])
    # filter_min = random_shear_rotation(filter_min)
    # filter_min = morphological_operations(filter_min)
    return sub_canvas, lu_pos, sub_canvas.shape, next_pos, True


def inference_sqrt_deletion(images, sizes, lu_poses, delete_ims):
    """
    inference sqrt subnode deletion

    Args:
        images: List[ndarray]
            inferenced elements inside the subnode
        sizes: List[List[int, int]]
            sizes of all the elements inside the subnode
        lu_poses: List[tuple[int, int]]
            left up position of the elements inside the subnode
        delete_ims: Dict[ndarray]
            deletion symbols dictionary
    Returns:

    """
    global_offset = lu_poses[0]
    min_x, min_y, max_x, max_y = cal_box_size(sizes, lu_poses)
    # total_w = lu_poses[-1][0] - lu_poses[0][0] + sizes[-1][1]
    total_w = max_x - min_x
    sqrt_canvas = np.ones((max_y - min_y + 10, total_w + 10)) * 255
    for i in range(len(images)):
        x = lu_poses[i][0] - global_offset[0]
        y = lu_poses[i][1] - global_offset[1]
        h, w = sizes[i]
        sqrt_canvas[y:y + h, x:x + w] = images[i]
    if len(images) - 1 < 2:
        delete_sample = random.choice(delete_ims['one_char']['sqrt'])
    elif len(images) - 1 < 5:
        delete_sample = random.choice(delete_ims['short']['sqrt'])
    else:
        delete_sample = random.choice(delete_ims['mid'])
    del_offset = delete_sample['offset']
    ratio = del_offset['ratio']
    con_h, con_w = sqrt_canvas.shape
    temp_ratio = [con_w / del_offset['w'], con_h / del_offset['h']]
    con_lu_pos = [0, 0]
    del_lu_pos = [-int(del_offset['left_up_w']*temp_ratio[0]), -int(del_offset['left_up_h']*temp_ratio[1])]
    # del_lu_pos = [0, 0]
    min_x = min(con_lu_pos[0], del_lu_pos[0])
    min_y = min(con_lu_pos[1], del_lu_pos[1])
    con_lu_pos = [int(con_lu_pos[0] + abs(min_x)), int(con_lu_pos[1] + abs(min_y))]
    del_lu_pos = [int((del_lu_pos[0] + abs(min_x))), int((del_lu_pos[1] + abs(min_y)))]
    con_rd_pos = [con_lu_pos[0] + con_w, con_lu_pos[1] + con_h]
    del_rd_pos = [con_rd_pos[0] - int(del_offset['right_down_w'] * temp_ratio[0]),
                  con_rd_pos[1] - int(del_offset['right_down_h'] * temp_ratio[1])]
    delete_sample_img = cv2.resize(delete_sample['img'], (del_rd_pos[0] - del_lu_pos[0], del_rd_pos[1] - del_lu_pos[1]))
    del_h, del_w = delete_sample_img.shape

    sub_canvas_w = max(con_lu_pos[0] + con_w, del_lu_pos[0] + del_w)
    sub_canvas_h = max(con_lu_pos[1] + con_h, del_lu_pos[1] + del_h)
    sub_canvas = np.ones((sub_canvas_h, sub_canvas_w)) * 255
    # pdb.set_trace()
    sub_canvas[con_lu_pos[1]: con_lu_pos[1] + con_h, con_lu_pos[0]: con_lu_pos[0] + con_w] = sqrt_canvas
    sub_canvas[del_lu_pos[1]: del_lu_pos[1] + del_h, del_lu_pos[0]: del_lu_pos[0] + del_w] = np.minimum(
        sub_canvas[del_lu_pos[1]: del_lu_pos[1] + del_h, del_lu_pos[0]: del_lu_pos[0] + del_w],
        delete_sample_img)
    return sub_canvas, sub_canvas.shape, lu_poses[0]


def inference_frac_deletion(images, sizes, lu_poses, delete_ims, box_size):
    """
    inference sqrt subnode deletion

    Args:
        images: List[ndarray]
            inferenced elements inside the subnode
        sizes: List[List[int, int]]
            sizes of all the elements inside the subnode
        lu_poses: List[tuple[int, int]]
            left up position of the elements inside the subnode
        delete_ims: Dict[ndarray]
            deletion symbols dictionary
    Returns:

    """
    global_offset = box_size[0:2]
    # pdb.set_trace()
    total_w = lu_poses[-1][0] - global_offset[0] + sizes[-1][1]
    frac_canvas = np.ones(((box_size[3] - box_size[1]) + 10, total_w + 10)) * 255
    for i in range(len(images)):
        x = lu_poses[i][0] - global_offset[0]
        y = lu_poses[i][1] - global_offset[1]
        h, w = images[i].shape
        frac_canvas[y:y + h, x:x + w] = images[i]
    if len(images) - 1 < 4:
        delete_sample = random.choice(delete_ims['one_char']['frac'])
    elif len(images) - 1 < 7:
        delete_sample = random.choice(delete_ims['short']['frac'])
    else:
        delete_sample = random.choice(delete_ims['mid'])
    del_offset = delete_sample['offset']
    ratio = del_offset['ratio']
    con_h, con_w = frac_canvas.shape
    temp_ratio = [con_w / del_offset['w'], con_h / del_offset['h']]
    con_lu_pos = [0, 0]
    del_lu_pos = [-int(del_offset['left_up_w']*temp_ratio[0]), -int(del_offset['left_up_h']*temp_ratio[1])]
    # del_lu_pos = [0, 0]
    min_x = min(con_lu_pos[0], del_lu_pos[0])
    min_y = min(con_lu_pos[1], del_lu_pos[1])
    con_lu_pos = [int(con_lu_pos[0] + abs(min_x)), int(con_lu_pos[1] + abs(min_y))]
    del_lu_pos = [int((del_lu_pos[0] + abs(min_x))), int((del_lu_pos[1] + abs(min_y)))]
    con_rd_pos = [con_lu_pos[0] + con_w, con_lu_pos[1] + con_h]
    del_rd_pos = [con_rd_pos[0] - int(del_offset['right_down_w']*temp_ratio[0]), con_rd_pos[1] - int(del_offset['right_down_h']*temp_ratio[1])]
    delete_sample_img = cv2.resize(delete_sample['img'], (del_rd_pos[0] - del_lu_pos[0], del_rd_pos[1] - del_lu_pos[1]))
    del_h, del_w = delete_sample_img.shape

    sub_canvas_w = max(con_lu_pos[0] + con_w, del_lu_pos[0] + del_w)
    sub_canvas_h = max(con_lu_pos[1] + con_h, del_lu_pos[1] + del_h)
    sub_canvas = np.ones((sub_canvas_h, sub_canvas_w)) * 255
    # pdb.set_trace()
    sub_canvas[con_lu_pos[1]: con_lu_pos[1] + con_h, con_lu_pos[0]: con_lu_pos[0] + con_w] = frac_canvas
    sub_canvas[del_lu_pos[1]: del_lu_pos[1] + del_h, del_lu_pos[0]: del_lu_pos[0] + del_w] = np.minimum(
        sub_canvas[del_lu_pos[1]: del_lu_pos[1] + del_h, del_lu_pos[0]: del_lu_pos[0] + del_w],
        delete_sample_img)


    # delete_sample = cv2.resize(delete_sample, (frac_canvas.shape[1], frac_canvas.shape[0]))
    # filter_min = np.minimum(delete_sample, frac_canvas)

    return sub_canvas, sub_canvas.shape, global_offset


def cal_leaf_nodes(node, num_nodes):
    """
    calculates the number of leaf nodes in a sub tree
    """
    for sub_node in node:
        try:
            sub_node = sub_node[0]
        except:
            pass
        if sub_node['type'] == 'leaf' or sub_node['type'] == 'deletion':  #
            num_nodes += 1
        elif sub_node['type'] == 'frac':
            num_nodes = cal_leaf_nodes(sub_node['num'][0], num_nodes)
            num_nodes = cal_leaf_nodes(sub_node['den'][0], num_nodes)
            num_nodes += 1
        elif sub_node['type'] == 'sqrt':
            num_nodes = cal_leaf_nodes(sub_node['content'][0], num_nodes)
            num_nodes += 2
            if sub_node['root_num']['num'] is not None:
                num_nodes += 1
        elif sub_node['type'] == 'scription':
            num_nodes = cal_leaf_nodes(sub_node['content'][0], num_nodes)
        elif sub_node['type'] == 'one_arg':
            num_nodes = cal_leaf_nodes(sub_node['content'][0], num_nodes)
    return num_nodes


def initialize_positions():
    min_x = math.inf
    min_y = math.inf
    max_x = 0
    max_y = 0
    return min_x, min_y, max_x, max_y


def cal_box_size(sizes, lu_poses):
    """
    calculates the box size of a subtree
    """
    min_x, min_y, max_x, max_y = initialize_positions()
    for ind in range(len(sizes)):
        # pdb.set_trace()
        min_x = min(min_x, lu_poses[ind][0])
        min_y = min(min_y, lu_poses[ind][1])
        max_x = max(max_x, lu_poses[ind][0] + sizes[ind][1])
        max_y = max(max_y, lu_poses[ind][1] + sizes[ind][0])
    return min_x, min_y, max_x, max_y


def combine_leaves(virtual_im):
    i = 0
    out_virtual_im = []
    while i < len(virtual_im):
        node = virtual_im[i]
        if node['type'] == 'leaf':
            if i < len(virtual_im) - 1 and virtual_im[i + 1]['type'] == 'leaf':
                leaves_node = {'type': 'leaves',
                               'children': [node, virtual_im[i + 1]]}
                i += 2
                while i < len(virtual_im) and virtual_im[i]['type'] == 'leaf':
                    leaves_node['children'].append(virtual_im[i])
                    i += 1
                i -= 1
                out_virtual_im.append(leaves_node)
            else:
                out_virtual_im.append(node)

        else:
            out_virtual_im.append(node)
        i += 1
        # print(i)
    return out_virtual_im


def inference_characters(virtual_im, canvas, characters_dict, crohme_path_dict, config, author, curr_pos, images,
                         lu_poses, sizes, cur_poses, label, offset_position, delete_ims, in_move, move_lu_poses,
                         move_sizes, move_imgs,
                         model, use_layout_model, in_den, rect_inds):
    """
    Recursively collects all the infomation to inference the handwritten image

    Args:
        virtual_im: List[Dict]
            It is a list of all the nodes
            please see src/util/define_nodes for all possible nodes
        canvas: Tuple
            canvas start and end points
        characters_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        crohme_path_dict: Dict
            Dict containing writter and the paths of charaters for the writer
        config: configuration file
        author: int
            selected author
        curr_pos: Tuple
            previous character end position
        images: List[Uint8Ndarray]
            character images
        lu_poses: List[List]
            left up positions of all the characters
        sizes: List[List]
            h and w of all the characters
        cur_poses: List[Tuple]
            end poses of all the characters
        label: List
            Modified label
        offset_position: str
            offset position up or middle
        delete_ims: List[Uint8Ndarray]
            all images that need to be deleted
        in_move: bool
            True if the current iteration is in move node
        move_lu_poses: List[List]
            left up positions of all move characters
        move_sizes: List[List]
            h and w of all the move characters
        move_imgs: List[Uint8Ndarray]
            all images that need to be moved

    Returns: sizes, lu positions and images to inference the image and canvas information
    """
    if in_move:
        try:
            temp_curr_pos = cur_poses[-1]
        except:
            temp_curr_pos = [0, 0]

    if use_layout_model:
        parts_virtual_im = combine_leaves(virtual_im)
    else:
        parts_virtual_im = virtual_im

    if type(parts_virtual_im) != list:
        parts_virtual_im = [parts_virtual_im]
    for ind in range(len(parts_virtual_im)):
        try:
            node = parts_virtual_im[ind]
        except:
            pdb.set_trace()
        prev_w = math.inf
        prev_h = math.inf
        try:
            node['type'] == 'leaf'
        except:
            node = node[0]
        if node['type'] == 'leaf' or node['type'] == 'deletion':
            if node['type'] == 'deletion':
                ims, lu_pos, size, next_pos, flag = inference_deletion(node, characters_dict, config, author,
                                                                       curr_pos, prev_h, prev_w, offset_position,
                                                                       delete_ims)
            else:
                if is_chinese(node['content']):
                    ims, lu_pos, size, next_pos, flag = inference_text(node, characters_dict, config, author,
                                                                       curr_pos, prev_h, prev_w, offset_position)
                else:
                    ims, lu_pos, size, next_pos, flag = inference_char(node, characters_dict, crohme_path_dict,
                                                                       config, author, curr_pos, prev_h, prev_w,
                                                                       offset_position, in_den)
            in_den = False
            label.append(node['label'])
            if in_move:
                temp_curr_pos = [temp_curr_pos[0] + node['actual_offset'][0],
                                 temp_curr_pos[1] + node['actual_offset'][1]]
                cur_poses.append(temp_curr_pos)
                move_lu_poses.append(lu_pos)
                move_sizes.append(size)
                move_imgs.append(ims)
                curr_pos = next_pos
            else:
                lu_poses.append(lu_pos)
                images.append(ims)
                sizes.append(size)
                curr_pos = next_pos
                cur_poses.append(curr_pos)
            if offset_position == 'middle':
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], next_pos[0]), max(canvas[1][1], next_pos[1] + size[0] // 2)))
            elif offset_position == 'up':
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], next_pos[0]), max(canvas[1][1], next_pos[1] + size[0])))

        elif node['type'] == 'leaves':
            cur_ims, cur_lu_poses, cur_sizes, cur_next_poses, flag = inference_leaves(node, canvas, characters_dict,
                                                                                      config, author,
                                                                                      curr_pos, prev_h, prev_w,
                                                                                      offset_position, model)

            if in_move:
                temp_curr_poses = [
                    [temp_curr_pos[0] + node['actual_offset'][0], temp_curr_pos[1] + node['actual_offset'][1]] for
                    temp_curr_pos in cur_next_poses]
                cur_poses.extend(temp_curr_poses)
                move_lu_poses.extend(cur_lu_poses)
                move_sizes.extend(cur_sizes)
                move_imgs.extend(cur_ims)
                curr_pos = temp_curr_poses[-1]
            else:
                lu_poses.extend(cur_lu_poses)
                images.extend(cur_ims)
                sizes.extend(cur_sizes)
                curr_pos = cur_next_poses[-1]
                cur_poses.extend(cur_next_poses)

            # print('cur_ims', len(cur_ims))
            # print('cur_lu_poses', len(cur_lu_poses))
            # print('cur_sizes', len(cur_sizes))
            # print('cur_next_poses', len(cur_next_poses))

            if offset_position == 'middle':
                for j in range(len(cur_ims)):
                    canvas = ((min(canvas[0][0], cur_lu_poses[j][0]), min(canvas[0][1], cur_lu_poses[j][1])),
                              (max(canvas[1][0], cur_next_poses[j][0]),
                               max(canvas[1][1], cur_next_poses[j][1] + cur_sizes[j][0] // 2)))
                    # print(j, len(lu_poses))
            elif offset_position == 'up':
                for j in range(len(cur_ims)):
                    canvas = ((min(canvas[0][0], cur_lu_poses[j][0]), min(canvas[0][1], cur_lu_poses[j][1])),
                              (max(canvas[1][0], cur_next_poses[j][0]),
                               max(canvas[1][1], cur_next_poses[j][1] + cur_sizes[j][0])))

        elif node['type'] == 'frac':
            label.append('\\frac')
            label.append('{')
            # pdb.set_trace()
            num_offset = node['num_offset']
            den_offset = node['den_offset']
            global_offset = node['global_offset']
            shift_x = random.randint(-math.ceil(abs(global_offset[0]) * 0.5), 0)
            shift_y = random.randint(-math.ceil(abs(global_offset[1]) * 0.1),
                                     math.ceil(abs(global_offset[1]) * 0.1))
            curr_pos = (curr_pos[0] + global_offset[0], curr_pos[1] + global_offset[1])
            in_den = True
            # numerator
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                node['num'][0], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, in_move, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model, in_den, rect_inds)
            label.append('}')
            label.append('{')
            num_nodes = cal_leaf_nodes(node['num'][0], 0)
            min_x, min_y, max_x, max_y = cal_box_size(sizes[len(sizes) - num_nodes:],
                                                      lu_poses[len(lu_poses) - num_nodes:])
            # pdb.set_trace()
            rect_lu_pos = (
                abs((num_offset[1][0] - num_offset[0][0]) - min_x), max_y + (num_offset[0][1] - num_offset[1][1]))
            # shift_x = random.randint(-math.ceil(abs(num_offset[0][0] - num_offset[1][0])), -math.ceil(abs(num_offset[0][0] - num_offset[1][0]) * 0.5))
            shift_y = random.randint(-math.ceil(abs(num_offset[0][1] - num_offset[1][1]) * 0.9),
                                     -math.ceil(abs(num_offset[0][1] - num_offset[1][1]) * 0.8))
            rect_lu_pos = (rect_lu_pos[0], rect_lu_pos[1] + shift_y)
            curr_pos = (rect_lu_pos[0], rect_lu_pos[1] + 10)
            num_nodes += cal_leaf_nodes(node['den'][0], 0)
            # denominator
            in_den = True
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                node['den'][0], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, in_move, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model, in_den, rect_inds)
            label.append('}')
            in_den = False
            min_x, min_y, max_x, max_y = cal_box_size(sizes[len(sizes) - num_nodes:],
                                                      lu_poses[len(lu_poses) - num_nodes:])
            rect_inds.append(len(sizes))
            ims, size, flag = inference_rect(node['line_sym'], canvas, characters_dict, crohme_path_dict, config,
                                             author, (max_x - min_x))
            lu_poses.append(rect_lu_pos)
            images.append(ims)
            sizes.append(size)
            min_x, min_y, max_x, max_y = cal_box_size(sizes[len(sizes) - num_nodes - 1:],
                                                      lu_poses[len(lu_poses) - num_nodes - 1:])
            curr_pos = [max_x, min_y]
            cur_poses.append(curr_pos)
            canvas = ((min(canvas[0][0], min_x), min(canvas[0][1], min_y)),
                      (max(canvas[1][0], max_x), max(canvas[1][1], max_y)))

        elif node['type'] == 'sqrt':
            label.append('\\sqrt')
            sqrt_sym = node['sqrt_sym']['sym']
            offset = node['sqrt_sym']['offset']
            temp_node = {'content': sqrt_sym,
                         'offset': offset,
                         'h': node['sqrt_sym']['h'],
                         'w': node['sqrt_sym']['w'],
                         'scale': None}
            surd_ind = len(sizes)
            ims, lu_pos, size, next_pos, flag = inference_char(temp_node, characters_dict, crohme_path_dict,
                                                               config, author, curr_pos, math.inf, math.inf,
                                                               offset_position, in_den)
            lu_poses.append(lu_pos)
            images.append(ims)
            sizes.append(size)
            if node['root_num']['num'] is not None:
                label.append('[')
                temp_node = {'content': node['root_num']['num'],
                             'offset': node['root_num']['offset'],
                             'h': math.ceil(node['root_num']['h'] * 0.7),
                             'w': math.ceil(node['root_num']['w'] * 0.9),
                             'scale': None}
                ims, lu_pos, size, _, flag = inference_char(temp_node, characters_dict, crohme_path_dict,
                                                            config, author, curr_pos, math.inf, math.inf,
                                                            offset_position, in_den)
                lu_poses.append(lu_pos)
                images.append(ims)
                sizes.append(size)
                label.append(node['root_num']['label'])
                label.append(']')
            curr_pos = next_pos

            if offset_position == 'middle':
                rect_pos = (curr_pos[0], curr_pos[1] - sizes[surd_ind][0] // 2)
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], curr_pos[0]), max(canvas[1][1], curr_pos[1] + size[0] // 2)))
            elif offset_position == 'up':
                rect_pos = curr_pos
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], curr_pos[0]), max(canvas[1][1], curr_pos[1] + size[0])))
            num_nodes = cal_leaf_nodes(node['content'][0], 0)
            label.append('{')
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                node['content'][0], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, in_move, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model, in_den, rect_inds)
            label.append('}')
            min_x, min_y, max_x, max_y = cal_box_size(sizes[len(sizes) - num_nodes:],
                                                      lu_poses[len(lu_poses) - num_nodes:])
            rect_inds.append(len(sizes))
            ims, size, flag = inference_rect(node['line_sym'], canvas, characters_dict, crohme_path_dict, config,
                                             author, (max_x - min_x))
            lu_poses.append(rect_pos)
            images.append(ims)
            sizes.append(size)
            # Resize surd based on the max of height of the content
            # sizes[surd_ind][0] = max_y - lu_poses[surd_ind][1]
            # images[surd_ind] = cv2.resize(images[surd_ind], (images[surd_ind].shape[1], sizes[surd_ind][0]))
            canvas = ((min(canvas[0][0], min_x), min(canvas[0][1], min_y)),
                      (max(canvas[1][0], max_x), max(canvas[1][1], max_y)))
            curr_pos = (rect_pos[0] + size[1], rect_pos[1])
            cur_poses.append(curr_pos)

        elif node['type'] == 'scription':
            label.append(node['label'])
            label.append('{')
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                node['content'][0], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, in_move, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model, in_den, rect_inds)
            label.append('}')
            curr_pos = cur_poses[-1]

        elif node['type'] == 'one_arg':
            left_offset = node['left_offset']
            right_offset = node['right_offset']
            global_offset = node['global_offset']
            curr_pos = (curr_pos[0] + global_offset[0], curr_pos[1] + global_offset[1])
            label.append(node['special_sym']['label'])
            label.append('{')
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                node['content'][0], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, in_move, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model, in_den, rect_inds)
            label.append('}')
            spec_sym = node['special_sym']['sym']
            offset = node['special_sym']['offset']
            temp_node = {'content': spec_sym,
                         'offset': offset,
                         'h': node['special_sym']['h'],
                         'w': node['special_sym']['w'],
                         'scale': None}
            ims, lu_pos, size, next_pos, flag = inference_char(temp_node, characters_dict, crohme_path_dict,
                                                               config, author, curr_pos, prev_h, prev_w,
                                                               offset_position, in_den)
            lu_poses.append(lu_pos)
            images.append(ims)
            sizes.append(size)
            num_nodes = cal_leaf_nodes(node['content'][0], 0) + 1
            min_x, min_y, max_x, max_y = cal_box_size(sizes[len(sizes) - num_nodes:],
                                                      lu_poses[len(lu_poses) - num_nodes:])
            canvas = ((min(canvas[0][0], min_x), min(canvas[0][1], min_y)),
                      (max(canvas[1][0], max_x), max(canvas[1][1], max_y)))
            curr_pos = (max_x, min_y)
            cur_poses.append(curr_pos)

        elif node['type'] == 'sqrt_deletion':
            # pdb.set_trace()
            temp_label = label.copy()
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                [node['content']], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, False, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model,
                in_den, rect_inds)
            num_elements = cal_leaf_nodes([node['content']], 0)
            img, size, lu_pos = inference_sqrt_deletion(images[-num_elements:], sizes[-num_elements:],
                                                        lu_poses[-num_elements:], delete_ims)
            del sizes[-num_elements:]
            del images[-num_elements:]
            del lu_poses[-num_elements:]
            label = temp_label
            label.append('\\deletion')
            images.append(img)
            sizes.append(size)
            lu_poses.append(lu_pos)
            if offset_position == 'middle':
                curr_pos = (lu_pos[0] + size[1], lu_pos[1] + size[0] // 2)
            else:
                curr_pos = (lu_pos[0] + size[1], lu_pos[1])
            cur_poses.append(curr_pos)
            if offset_position == 'middle':
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], curr_pos[0]), max(canvas[1][1], curr_pos[1] + size[0] // 2)))
            elif offset_position == 'up':
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], curr_pos[0]), max(canvas[1][1], curr_pos[1] + size[0])))

        elif node['type'] == 'frac_deletion':
            # pdb.set_trace()
            temp_label = label.copy()
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                [node['content']], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, False, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model,
                in_den, rect_inds)
            num_elements = cal_leaf_nodes([node['content']], 0)
            min_x, min_y, max_x, max_y = cal_box_size(sizes[-num_elements:],
                                                      lu_poses[-num_elements:])
            img, size, lu_pos = inference_frac_deletion(images[-num_elements:], sizes[-num_elements:],
                                                        lu_poses[-num_elements:], delete_ims,
                                                        [min_x, min_y, max_x, max_y])
            del sizes[-num_elements:]
            del images[-num_elements:]
            del lu_poses[-num_elements:]
            label = temp_label
            label.append('\\deletion')
            images.append(img)
            sizes.append(size)
            lu_poses.append(lu_pos)
            if offset_position == 'middle':
                curr_pos = (lu_pos[0] + size[1], lu_pos[1] + size[0] // 2)
            else:
                curr_pos = (lu_pos[0] + size[1], lu_pos[1])
            cur_poses.append(curr_pos)
            if offset_position == 'middle':
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], curr_pos[0]), max(canvas[1][1], curr_pos[1] + size[0] // 2)))
            elif offset_position == 'up':
                canvas = ((min(canvas[0][0], lu_pos[0]), min(canvas[0][1], lu_pos[1])),
                          (max(canvas[1][0], curr_pos[0]), max(canvas[1][1], curr_pos[1] + size[0])))


        elif node['type'] == 'move':
            prev_curr_pos = curr_pos
            curr_pos = (curr_pos[0] + node['offset'][0], curr_pos[1] + node['offset'][1])
            temp_node = []
            for sub_node in node['content'][0]:
                sub_node['scale'] = 0.9
                temp_node.append(sub_node)
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                temp_node, canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, True, move_lu_poses, move_sizes, move_imgs, model, use_layout_model,
                in_den, rect_inds)
            num_elements = cal_leaf_nodes(node['content'], 0)
            min_x, min_y, max_x, max_y = cal_box_size(sizes[-num_elements:],
                                                      lu_poses[-num_elements:])
            canvas = ((min(canvas[0][0], min_x), min(canvas[0][1], min_y)),
                      (max(canvas[1][0], max_x), max(canvas[1][1], max_y)))
            in_move = False
            del cur_poses[-len(node['content'][0]):]
            curr_pos = cur_poses[-1]

        elif node['type'] == 'sqrt_move':
            prev_curr_pos = curr_pos
            curr_pos = (curr_pos[0] + node['offset'][0], curr_pos[1] + node['offset'][1])
            # temp_node = []
            # for sub_node in node['content'][0]:
            #     sub_node['scale'] = 0.9
            #     temp_node.append(sub_node)
            # pdb.set_trace()
            images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
                node['content'][0], canvas, characters_dict,
                crohme_path_dict, config, author, curr_pos, images, lu_poses, sizes, cur_poses,
                label, offset_position, delete_ims, False, move_lu_poses, move_sizes, move_imgs, model,
                use_layout_model,
                in_den, rect_inds)
            num_elements = cal_leaf_nodes(node['content'], 0)
            min_x, min_y, max_x, max_y = cal_box_size(sizes[-num_elements:],
                                                      lu_poses[-num_elements:])
            canvas = ((min(canvas[0][0], min_x), min(canvas[0][1], min_y)),
                      (max(canvas[1][0], max_x), max(canvas[1][1], max_y)))
            in_move = False
            del cur_poses[-num_elements:]
            curr_pos = cur_poses[-1]
    return images, lu_poses, sizes, canvas, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds


def inference_image(virtual_im, characters_dict, crohme_path_dict, delete_ims,
                    config, incomplete_deletion_flag, id, offset_position, author, use_layout_model):
    """
    given meta information about a image, inference the image

    Args:
        virtual_im: obj

    Returns:
        inferenced_im: uint8ndarray
        inferenced image
        modified label
    """
    if use_layout_model:
        model = load_model()
    else:
        model = None
    curr_pos = (0, 0)
    canvas = ((0, 0), (0, 0))
    images, lu_poses, sizes, canvas_shape, cur_poses, label, move_lu_poses, move_sizes, move_imgs, rect_inds = inference_characters(
        virtual_im, canvas, characters_dict, crohme_path_dict,
        config, author, curr_pos, [], [], [], [], [], offset_position, delete_ims,
        False, [], [], [], model, config.USE_LAYOUT_MODEL, False, [])
    if images is None:
        return
    canvas = np.ones((canvas_shape[1][1] - canvas_shape[0][1] + 10, canvas_shape[1][0] - canvas_shape[0][0] + 20)) * 255
    min_x = canvas_shape[0][0]
    min_y = canvas_shape[0][1]
    for ind in range(len(sizes)):
        lu_pos = (int(lu_poses[ind][0] + abs(min_x)), int(lu_poses[ind][1] + abs(min_y)))
        size = sizes[ind]
        im = images[ind]
        canvas[lu_pos[1]:lu_pos[1] + size[0], lu_pos[0]:lu_pos[0] + size[1]] = np.minimum(
            canvas[lu_pos[1]:lu_pos[1] + size[0], lu_pos[0]:lu_pos[0] + size[1]], im)
    if move_lu_poses != []:
        for ind in range(len(move_lu_poses)):
            lu_pos = (int(move_lu_poses[ind][0] + abs(min_x)), int(move_lu_poses[ind][1] + abs(min_y)))
            size = move_sizes[ind]
            im = move_imgs[ind]
            canvas[lu_pos[1]:lu_pos[1] + size[0], lu_pos[0]:lu_pos[0] + size[1]] = np.minimum(
                canvas[lu_pos[1]:lu_pos[1] + size[0], lu_pos[0]:lu_pos[0] + size[1]], im)
    # canvas = morphological_operations(canvas)
    canvas = deskew(canvas)
    canvas = 255 - fix_lr_boundary(255 - canvas)
    if '\\deletion' in label:
        cv2.imwrite(os.path.join(config.DELETE_OUT_PATH, id + '_del.png'), canvas)
    cv2.imwrite(os.path.join(config.OUT_PATH, id + '.png'), canvas)
    return canvas, ' '.join(label)
