from math import floor
from typing import Dict, List, Any, Union

import numpy as np
import cv2
import random
import pdb

from src.util.image_util import flip_tuple, blur_image
from src.util.util import cal_leaf_nodes
from src.util.define_nodes import nodes
from src.util.misc_util import is_chinese

empty_node = nodes()
remove_elements = ['{', '}', '^', '_', '[', ']']

def incomplete_deletion(text, casia_character_dict, delete_ims, h, w, image):
    """
    Given symbol and strokes simulate incomplete deletion

    params:
        symbol:
        casia_character_dict: 
            key: symbol 
            value: stokes, max_h, max_w
    return:
        img: ndarray
    """
    symbol = text['id']
    print(symbol)
    strokes = casia_character_dict[symbol]['strokes']
    max_h = casia_character_dict[symbol]['h']
    max_w = casia_character_dict[symbol]['w']
    goal_w = 100
    ratio = goal_w / max_w
    goal_h = floor(ratio * max_h)
    canvas = np.ones((goal_h + 1, goal_w + 1), dtype=np.uint8) * 255
    rand_strokes = random.randint(1, len(strokes) - 1)
    for ind in range(rand_strokes):
        stroke_arr = np.asarray(strokes[ind])
        for j in range(len(stroke_arr) - 1):
            cv2.line(canvas, flip_tuple(stroke_arr[j]), flip_tuple(stroke_arr[j + 1]), 0, 2)
    canvas = blur_image(canvas)
    box_h = int(text['h'])
    box_w = int(text['w'] - 30)
    lu_position = text['lu_position']
    if lu_position[0] + box_w > w:
        box_w = box_w - (lu_position[0] + box_w - w)
    delete_sample = delete_ims['alphabet'][5]
    canvas = cv2.resize(canvas, (box_w, box_h))
    delete_sample = cv2.resize(delete_sample, (box_w, box_h))
    canvas = np.minimum(image[lu_position[1]:lu_position[1] + box_h, lu_position[0]:lu_position[0] + box_w], canvas)
    canvas = np.minimum(delete_sample, canvas)
    cv2.imwrite('temp.png', canvas)
    return canvas, box_w, box_h


def process_leaf(node, prev_del, in_move):
    # pdb.set_trace()
    element = empty_node.leaf_node()
    element['content'] = node['content']
    if prev_del:
        element['offset'] = [random.randint(5, 7), random.randint(0, 3)]
    elif in_move:
        element['offset'] = [random.randint(5, 7), random.randint(0, 3)]
    else:
        element['offset'] = node['offset']
    element['actual_offset'] = node['offset']
    element['h'] = node['h']
    element['w'] = node['w']
    element['scale'] = node['scale']
    element['label'] = node['label']
    return element


def pick_delete_character(deleted_char, writter_dict, chinese_set, math_set):
    writter_set = set(writter_dict)
    delete_char_len = random.choices(population=[1, 2, 3, 4],
                                     weights=[0.9, 0.09, 0.009, 0.001],
                                     k=1)[0]
    if len(deleted_char) == 1 and is_chinese(deleted_char):
        return [random.choice(tuple(chinese_set.intersection(writter_set))) for _ in range(delete_char_len)]
    else:
        return [random.choice(tuple(math_set.intersection(writter_set))) for _ in range(delete_char_len)]


def process_del(node, chinese_symbols, math_symbols, character_dict, author):
    element = empty_node.deletion_node()
    # pdb.set_trace()
    if node['type'] == 'leaf':
        element['content'] = pick_delete_character(node['content'], character_dict[author], set(chinese_symbols),
                                                   set(math_symbols))
    elif node['type'] == 'sqrt':
        element['content'] = node
    element['offset'] = node['offset']
    element['scale'] = node['scale']
    element['h'] = node['h']
    element['w'] = node['w']
    element['label'] = '\\deletion'
    return element


def cal_consecutive_leafs(virtual_image):
    leaf_pos = []
    for i in range(len(virtual_image)):
        try:
            virtual_image[i]['type']
        except:
            virtual_image[i] = virtual_image[i][0]
        if virtual_image[i]['type'] == 'leaf':
            leaf_pos.append(i)
        else:
            if leaf_pos != [] and virtual_image[i]['type'] == 'scription':
                del leaf_pos[-1]
    return leaf_pos


def augment_deletion(virtual_image, del_poses, new_tree, global_pos, chinese_symbols, math_symbols, character_dict,
                     author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node):
    """
    adding deletion

    Args:
        move_sqrt_node:
        sqrt_del_pos:
        frac_del_pos:
        virtual_image: List[Dict]
        del_poses:
        new_tree:
        global_pos:
        chinese_symbols:
        math_symbols:
        character_dict:
        author:
        move_loc:
        moved_poses:
        in_move:
        ignore_char:

    Returns:

    """
    prev_del = False
    img_pos = 0
    leaf_poses = cal_consecutive_leafs(virtual_image)
    # pdb.set_trace()
    leaf_range = set([num + global_pos for num in leaf_poses])
    while img_pos < len(virtual_image):
        node = virtual_image[img_pos]
        if sqrt_del_pos is not None and sqrt_del_pos == img_pos:
            element = empty_node.sqrt_deletion_node()
            element['content'] = node
            sqrt_del_pos = None
        elif frac_del_pos is not None and frac_del_pos == img_pos:
            element = empty_node.frac_deletion_node()
            element['content'] = node
            frac_del_pos = None
        elif move_sqrt_node is not None and move_sqrt_node == img_pos:
            element = empty_node.sqrt_move_node()
            element['move_loc'] = move_loc
            # pdb.set_trace()
            if move_loc == 'up':
                x = int(random.uniform(-node['line_w'] * 1.2, -node['line_w'] * 1.3))
                y = int(random.uniform(-node['sqrt_sym']['h'] * 1.3, -node['sqrt_sym']['h'] * 1))
            else:
                x = int(random.uniform(-node['line_w'] * 1.2, -node['line_w'] * 1.3))
                y = int(random.uniform(node['sqrt_sym']['h'] * 1.3, node['sqrt_sym']['h'] * 1))
            element['offset'] = [x, y]
            num_elements = cal_leaf_nodes([node], 0)
            global_pos += num_elements
            move_sqrt_node = None
            element['content'].append(
                augment_deletion([node], del_poses, [], global_pos,
                                 chinese_symbols, math_symbols,
                                 character_dict, author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos,
                                 frac_del_pos, move_sqrt_node))
            img_pos += num_elements
            # in_move = False
            # pdb.set_trace()
            prev_del = False
        else:
            if node['type'] == 'frac':
                element: Dict[str, Union[str, List[int], None, List[Any]]] = empty_node.frac_node()
                element['num_offset'] = node['num_offset']
                element['den_offset'] = node['den_offset']
                element['global_offset'] = node['global_offset']
                element['line_sym'] = node['line_sym']
                move_loc = 'up'
                element['num'].append(
                    augment_deletion(node['num'][0], del_poses, [], global_pos, chinese_symbols, math_symbols,
                                     character_dict, author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node))
                global_pos += len(node['num'][0])
                move_loc = 'down'
                element['den'].append(
                    augment_deletion(node['den'][0], del_poses, [], global_pos, chinese_symbols, math_symbols,
                                     character_dict, author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node))
                img_pos += 1
                global_pos += len(node['den'][0])

            elif node['type'] == 'sqrt':
                element = empty_node.sqrt_node()
                element['line_sym'] = node['line_sym']
                element['root_num']['num'] = node['root_num']['num']
                element['root_num']['offset'] = node['root_num']['offset']
                element['root_num']['h'] = node['root_num']['h']
                element['root_num']['w'] = node['root_num']['w']
                element['root_num']['label'] = node['root_num']['label']
                element['sqrt_sym']['sym'] = node['sqrt_sym']['sym']
                element['sqrt_sym']['offset'] = node['sqrt_sym']['offset']
                element['sqrt_sym']['h'] = node['sqrt_sym']['h']
                element['sqrt_sym']['w'] = node['sqrt_sym']['w']
                move_loc = 'down'
                element['content'].append(
                    augment_deletion(node['content'][0], del_poses, [], global_pos, chinese_symbols, math_symbols,
                                     character_dict, author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node))
                img_pos += 1
                global_pos += len(node['content'][0])

            elif node['type'] == 'scription':
                element = empty_node.scription_node()
                element['label'] = node['label']
                if node['label'] == '^':
                    move_loc = 'up'
                else:
                    move_loc = 'down'
                element['content'].append(
                    augment_deletion(node['content'][0], del_poses, [], global_pos, chinese_symbols, math_symbols,
                                     character_dict, author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node))
                img_pos += 1
                global_pos += len(node['content'][0])

            elif node['type'] == 'one_arg':
                element = empty_node.one_arg_node()
                element['special_sym']['sym'] = node['special_sym']['sym']
                element['special_sym']['offset'] = node['special_sym']['offset']
                element['special_sym']['h'] = node['special_sym']['h']
                element['special_sym']['w'] = node['special_sym']['w']
                element['special_sym']['label'] = node['special_sym']['label']
                element['left_offset'] = node['left_offset']
                element['right_offset'] = node['right_offset']
                element['global_offset'] = node['global_offset']
                move_loc = 'down'
                element['content'].append(
                    augment_deletion(node['content'][0], del_poses, [], global_pos, chinese_symbols, math_symbols,
                                     character_dict, author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node))
                img_pos += 1
                global_pos += len(node['content'][0])

            else:
                if del_poses != [] and global_pos == del_poses[0]:
                    if node['content'].split('-')[-1] not in ignore_char:
                        element = process_del(node, chinese_symbols, math_symbols, character_dict, author)
                        prev_del = True
                    else:
                        element = process_leaf(node, prev_del, in_move)
                        in_move = False
                        prev_del = False
                        img_pos += 1
                        global_pos += 1
                    del del_poses[0]
                elif moved_poses != [] and global_pos == moved_poses[0][0]:
                    element = empty_node.move_node()
                    num_elements = len(set(moved_poses[0]).intersection(leaf_range))
                    if num_elements == 0:
                        del moved_poses[0]
                        continue
                    in_move = True
                    element['move_loc'] = move_loc
                    node['w'] = node['w'] * 0.9
                    node['h'] = node['h'] * 0.9
                    if move_loc == 'up':
                        x = int(random.uniform(-node['w'] * 1.2, -node['w'] * 1.5))
                        y = int(random.uniform(-node['h'] * 1.3, -node['h'] * 1))
                    else:
                        x = int(random.uniform(-node['w'] * 1.2, -node['w'] * 1.5))
                        y = int(random.uniform(node['h'] * 1.3, node['h'] * 1))
                    element['offset'] = [x, y]
                    global_pos += num_elements
                    element['content'].append(
                        augment_deletion(virtual_image[img_pos:img_pos + num_elements], del_poses, [], global_pos,
                                         chinese_symbols, math_symbols,
                                         character_dict, author, move_loc, moved_poses, in_move, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node))
                    img_pos += num_elements
                    del moved_poses[0]
                    in_move = False
                    prev_del = False
                else:
                    element = process_leaf(node, prev_del, in_move)
                    in_move = False
                    prev_del = False
                    img_pos += 1
                    global_pos += 1
        new_tree.append(element)
    return new_tree


def add_deletion(virtual_image, chinese_symbols, math_symbols, character_dict, author, ignore_char):
    ret_tree = []
    sqrt_del_pos = None
    frac_del_pos = None
    move_sqrt_node = None
    num_leaf_nodes = cal_leaf_nodes(virtual_image, 0)
    n_deletion = random.choices(population=[0, 1, 2, 3],
                                weights=[0.9, 0.09, 0.009, 0.001],
                                k=1)[0]
    # n_deletion = 1
    if n_deletion == 0:
        return virtual_image
    try:
        node_types = [node['type'] for node in virtual_image]
    except:
        pdb.set_trace()
    if 'sqrt' in node_types:
        if random.random() < 0.05:
            sqrt_del_pos = node_types.index('sqrt')
            if random.random() < 0.00005:
                move_sqrt_node = node_types.index('sqrt')
        pass
    elif 'frac' in node_types:
        if random.random() < 0.000005:
            frac_del_pos = node_types.index('frac')
    move_loc = random.choice(['up', 'down'])
    deletion_positions = sorted(random.sample(list(range(num_leaf_nodes)), k=n_deletion))
    # deletion_positions = []
    moved_poses = []
    for ind in range(len(deletion_positions)):
        if ind == 0 or deletion_positions[ind] != deletion_positions[ind - 1] + 1:
            if random.random() < 0.2:
                moved_chars_len = random.choices(population=[1, 2, 3, 4],
                                                 weights=[0.6, 0.39, 0.009, 0.001],
                                                 k=1)[0]
                # moved_chars_len = 1
                moved_pos = list(range(deletion_positions[ind], deletion_positions[ind] + moved_chars_len))
                moved_poses.append(moved_pos)
    ret_tree = augment_deletion(virtual_image, deletion_positions, ret_tree, 0, chinese_symbols, math_symbols,
                                character_dict, author, move_loc, moved_poses, False, ignore_char, sqrt_del_pos, frac_del_pos, move_sqrt_node)
    return ret_tree
