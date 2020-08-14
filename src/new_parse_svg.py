import math

import bs4
import re
import cv2
from os.path import join
import json
import numpy as np
from svgpathtools import svg2paths
from svg.path import parse_path
from PIL import ImageFont, ImageDraw, Image, ImageChops

from src.util.define_nodes import nodes
from src.util.util import unicode_to_symbol
import pdb

leaf_node_list = ['mi', 'mo', 'mn']
unsupported_characetrs = ['\\RomanNumeralCaps', '\\textperthousand', '\\romannumeral', '\\RomanNumeralCaps',
                          '\\textcircled']
node = nodes()
r1 = re.compile(".*?\((.*?)\)")

remove_elements = ['{', '}']


def parse_svg(svg_str, path, config, label):
    '''
    from a svg string, generate a virtual_image struct

    :param svg_str: str
        svg string
    :param path: str
        path to the svg string file
    :param config: config file
        configuration file
    :param label: str
        label to be used
    :return: Dict[Dict[List]]
        keys: characters, texts, root_rect, frac_rect, canvas_h, canvas_w, img_id
        virtual image that has all the information's to inference the image
    '''

    svg2xml = bs4.BeautifulSoup(svg_str, 'lxml')
    scaling_factor = config.SCALING_FACTOR
    tree = extract_trajectories(label, svg2xml, scaling_factor, path, config)
    return tree


def extract_char(im):
    pixel = []
    for x in range(im.shape[1]):
        for y in range(im.shape[0]):
            pix = im[x, y]
            if pix == 0:
                pixel.append([x, y])
    min_x = min([i[0] for i in pixel]) - 2
    max_x = max([i[0] for i in pixel]) + 2
    min_y = min([i[1] for i in pixel]) - 2
    max_y = max([i[1] for i in pixel]) + 2
    crop_im = im[min_x:max_x, min_y:max_y]
    return crop_im


def get_box_info(path_string, offset_x, offset_y, h_offset, scale, scaling_factor, mask, math_font_Path, id):
    '''

    Args:
        path_string:
        offset_x:
        offset_y:
        h_offset:
        scale:
        scaling_factor:
        mask:

    Returns:

    '''
    id = id.split('-')[-1]
    try:
        encoded_char = unicode_to_symbol(id)
    except:
        encoded_char = id
    contour = []
    for e in path_string:
        contour.append([[e.start.real, e.start.imag]])
        contour.append([[e.end.real, e.end.imag]])
    contour = np.array(contour) * scale * scaling_factor
    contour[:, :, 0] = contour[:, :, 0] * 1 + contour[:, :, 1] * 0 + 0 + offset_x
    contour[:, :, 1] = contour[:, :, 0] * 0 + contour[:, :, 1] * (-1) + 0 - h_offset - offset_y
    contour = contour.astype(int)
    # cv2.drawContours(mask, [contour], -1, 0, -1)
    min_x, max_x = contour[:, :, 0].min(), contour[:, :, 0].max()
    min_y, max_y = contour[:, :, 1].min(), contour[:, :, 1].max()
    box_h = max_y-min_y
    box_w = max_x-min_x
    # cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, 2)
    # img_pil = Image.fromarray(mask.astype(np.uint8))
    blank = Image.new('L', (100, 100))
    blank = ImageChops.constant(blank, 255)
    draw = ImageDraw.Draw(blank)
    font_size = 75 #int(884 * scaling_factor * scale)
    font = ImageFont.truetype(math_font_Path, font_size)
    w, h = draw.textsize(encoded_char, font=font)
    draw.text(((100-w)/2, (100-h)/2), encoded_char, font=font, fill=0)
    # draw.rectangle([(offset_x, offset_y),
    #                 (offset_x + font_size, offset_y + font_size)], None, 255)
    temp = np.array(blank)
    black_im = extract_char(temp)
    black_im = cv2.resize(black_im, (box_w, box_h))
    if encoded_char == ':':
        min_x = min_x + 10
    mask[min_y:min_y+black_im.shape[0], min_x:min_x+black_im.shape[1]] = black_im
    return mask


def get_offsets(attributes):
    '''

    Args:
        attributes:

    Returns:

    '''
    offsets = r1.match(attributes['transform'])
    m1 = offsets.group(1)
    offset_x = float(m1.split(',')[0])
    offset_y = float(m1.split(',')[1])
    if 'scale' in attributes['transform']:
        scale = attributes['transform'].split()[-1]
        scale_off = r1.match(scale)
        scale_temp = float(scale_off.group(1))
    else:
        scale_temp = 1
    return offset_x, offset_y, scale_temp


def get_rect_info(rect_element, offset_x, offset_y, h_offset, scale, scaling_factor, mask):
    '''

    Args:
        rect_element:
        offset_x:
        offset_y:
        h_offset:
        scale:
        scaling_factor:
        mask:

    Returns:

    '''
    box = [float(rect_element['x']) * scale * scaling_factor + offset_x,
           float(rect_element['y']) * scale * scaling_factor + offset_y,
           (float(rect_element['x']) * scale + float(rect_element['width']) * scale) * scaling_factor + offset_x,
           (float(rect_element['y']) * scale - float(rect_element['height']) * scale) * scaling_factor + offset_y]
    box[1] = abs(box[1] + h_offset)
    box[3] = abs(box[3] + h_offset)
    box = [int(i) for i in box]
    cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), 0, -1)
    return mask


def get_text_info(text_element, offset_x, offset_y, h_offset, scale, scaling_factor, mask, chinese_font_path,
                  shift_char=False):
    '''

    Args:
        text_element:
        offset_x:
        offset_y:
        h_offset:
        scale:
        scaling_factor:
        img_pil:
        chinese_font_path:
        shift_char:

    Returns:

    '''
    h, w = mask.shape
    img_pil = Image.fromarray(mask.astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    font_size = int(884 * scaling_factor * scale)
    font = ImageFont.truetype(chinese_font_path, font_size)
    if font_size < 34:
        offset_x, offset_y = int(offset_x), int(offset_y - h_offset)
    else:
        offset_x, offset_y = int(offset_x), int(offset_y)
        # if shift_char:
        #     offset_y = int(offset_y - h_offset / 2)
    draw.text((offset_x, h//2 - font_size//2), text_element, font=font, fill=0)
    # draw.rectangle([(offset_x, offset_y),
    #                 (offset_x + font_size, offset_y + font_size)], None, 255)
    mask = np.array(img_pil)
    return mask


def parse_elements(label, paths, offset_x, offset_y, scale, scaling_factor, h_offset, path_id_pair, mask, in_sub,
                   in_frac, in_sqrt, in_one_arg, in_temp, chinese_font_path, math_font_Path):
    '''

    Args:
        tree:
        label:
        paths:
        offset_x:
        offset_y:
        scale:
        scaling_factor:
        h_offset:
        path_id_pair:
        mask:
        in_sub:
        in_frac:
        in_sqrt:
        in_one_arg:
        chinese_font_path:

    Returns:
    :param label:
    :param math_font_Path:

    '''
    ind = 0
    offset_x_sub = 0
    offset_y_sub = 0
    while ind < len(paths):
        svg_node = paths[ind]
        attr = svg_node.attrs

        if 'fill' in attr:
            if attr['fill'] == 'red':
                # drop the sample
                # print('invalid svg')
                raise Exception('INVALID SVG')

        if in_frac == 0 and in_sub == 0 and in_sqrt == 0 and in_one_arg == 0 and in_temp == 0:
            offset_x = 0
            offset_y = 0
            scale = 1

        if 'data-mml-node' in attr:
            if attr['data-mml-node'] == 'merror':
                raise Exception('delimeter error')
            if attr['data-mml-node'] in leaf_node_list:
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                        offset_x_temp = offset_x_sub * scaling_factor * scale + offset_x
                        offset_y_temp = offset_y_sub * scaling_factor * scale + offset_y
                        scale *= scale_sub
                else:
                    offset_x_temp = offset_x
                    offset_y_temp = offset_y

                char_done = False
                for ele in svg_node:
                    if 'transform' in ele.attrs:
                        if 'translate' in ele.attrs['transform']:
                            offset_x_sub, offset_y_sub, scale_sub = get_offsets(ele.attrs)
                            char_offset_x = offset_x_sub * scaling_factor * scale + offset_x_temp
                            char_offset_y = offset_y_sub * scaling_factor * scale + offset_y_temp
                            scale *= scale_sub
                    else:
                        char_offset_x = offset_x_temp
                        char_offset_y = offset_y_temp
                    if 'xlink:href' in ele.attrs:
                        id = ele.attrs['xlink:href'].replace('#', '')
                        path_string = path_id_pair[id]
                        if len(path_string) == 0:
                            continue
                        mask = get_box_info(path_string, char_offset_x, char_offset_y, h_offset,
                                            scale,
                                            scaling_factor, mask, math_font_Path, id)
                        char_done = True
                    else:
                        if '\\frac' in label:
                            shift_char = True
                        else:
                            shift_char = False
                        for tag in svg_node:
                            if 'transform' in tag.attrs:
                                if 'translate' in tag['transform']:
                                    translate = tag['transform'].split('m')[0]
                                    val = r1.match(translate).group(1)
                                    text = tag.string
                                    text_offset_x = float(
                                        val.split(',')[0]) * scaling_factor * scale + offset_x_temp
                                    text_offset_y = float(
                                        val.split(',')[1]) * scaling_factor * scale + offset_y_temp

                                else:
                                    text = tag.string
                                    text_offset_x = offset_x_temp
                                    text_offset_y = offset_y_temp

                            if tag.string is not None:
                                mask = get_text_info(text, text_offset_x, text_offset_y, h_offset,
                                                     scale,
                                                     scaling_factor, mask,
                                                     chinese_font_path, shift_char)
                            elif not char_done:
                                text = tag.attrs['xlink:href'].replace('#', '')
                                path_string = path_id_pair[text]
                                mask = get_box_info(path_string, text_offset_x, text_offset_y,
                                                    h_offset,
                                                    scale,
                                                    scaling_factor, mask, math_font_Path, text)

                            else:
                                continue
                        break

            elif attr['data-mml-node'] == 'mrow':
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                        offset_x += offset_x_sub * scale * scaling_factor
                        offset_y += offset_y_sub * scale * scaling_factor
                        scale *= scale_sub

            elif attr['data-mml-node'] == 'TeXAtom':
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                        offset_x += offset_x_sub * scaling_factor
                        offset_y += offset_y_sub * scaling_factor
                        scale *= scale_sub
                    if len(svg_node.find_all('g')) > 0:
                        in_temp += 1
                        sub_svg_node = svg_node.find_all('g')
                        mask = parse_elements(label, sub_svg_node,
                                              offset_x,
                                              offset_y, scale,
                                              scaling_factor, h_offset,
                                              path_id_pair,
                                              mask,
                                              in_sub, in_frac,
                                              in_sqrt, in_one_arg,
                                              in_temp,
                                              chinese_font_path,math_font_Path)
                        if 'transform' in attr:
                            if 'translate' in attr['transform']:
                                offset_x -= offset_x_sub * scale * scaling_factor
                                offset_y -= offset_y_sub * scale * scaling_factor
                        in_temp -= 1
                        ind += len(svg_node.find_all('g'))

            elif attr['data-mml-node'] == 'mover':
                element = node.one_arg_node()
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                        offset_x += offset_x_sub * scaling_factor
                        offset_y += offset_y_sub * scaling_factor
                        scale *= scale_sub
                in_one_arg += 1
                sub_svg_node = svg_node.find_all('g')

                mask = parse_elements([sub_svg_node[-1]], offset_x,
                                      offset_y,
                                      scale,
                                      scaling_factor, h_offset, path_id_pair, mask,
                                      in_sub,
                                      in_frac, in_sqrt,
                                      in_one_arg, in_temp, chinese_font_path, math_font_Path)
                mask = parse_elements(sub_svg_node[:-1], offset_x,
                                      offset_y, scale,
                                      scaling_factor, h_offset, path_id_pair, mask,
                                      in_sub,
                                      in_frac, in_sqrt,
                                      in_one_arg, in_temp, chinese_font_path, math_font_Path)


                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x -= offset_x_sub * scale * scaling_factor
                        offset_y -= offset_y_sub * scale * scaling_factor
                in_one_arg -= 1
                ind += len(svg_node.find_all('g'))

            elif attr['data-mml-node'] == 'mfrac':
                element = node.frac_node()
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                        offset_x += offset_x_sub * scale * scaling_factor
                        offset_y += offset_y_sub * scale * scaling_factor
                        scale *= scale_sub
                in_frac += 1
                sub_svg_node = svg_node.find_all('g')
                num_num_ele = len(sub_svg_node[0].find_all('g')) + 1
                # pdb.set_trace()
                mask = parse_elements(label, sub_svg_node[:num_num_ele],
                                      offset_x,
                                      offset_y, scale,
                                      scaling_factor, h_offset, path_id_pair, mask,
                                      in_sub,
                                      in_frac,
                                      in_sqrt,
                                      in_one_arg, in_temp, chinese_font_path, math_font_Path)
                rect_element = svg_node.find_all('rect')[-1]
                mask = get_rect_info(rect_element, offset_x, offset_y, h_offset, scale,
                                     scaling_factor, mask)
                mask = parse_elements(label, sub_svg_node[num_num_ele:],
                                      offset_x,
                                      offset_y, scale,
                                      scaling_factor, h_offset, path_id_pair, mask,
                                      in_sub,
                                      in_frac,
                                      in_sqrt,
                                      in_one_arg, in_temp, chinese_font_path, math_font_Path)


                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x -= offset_x_sub * scale * scaling_factor
                        offset_y -= offset_y_sub * scale * scaling_factor
                ind += len(svg_node.find_all('g'))
                in_frac -= 1

            elif attr['data-mml-node'] == 'msqrt' or attr['data-mml-node'] == 'mroot':
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                        offset_x += offset_x_sub * scale * scaling_factor
                        offset_y += offset_y_sub * scale * scaling_factor
                        scale *= scale_sub
                in_sqrt += 1
                sub_svg_node = svg_node.find_all('g')
                mask = parse_elements(label, [sub_svg_node[-1]],
                                      offset_x,
                                      offset_y, scale,
                                      scaling_factor, h_offset,
                                      path_id_pair, mask,
                                      in_sub,
                                      in_frac, in_sqrt,
                                      in_one_arg, in_temp,
                                      chinese_font_path, math_font_Path)

                if attr['data-mml-node'] == 'mroot':
                    mask = parse_elements(label, [sub_svg_node[-2]], offset_x,
                                          offset_y,
                                          scale,
                                          scaling_factor, h_offset, path_id_pair, mask,
                                          in_sub,
                                          in_frac, in_sqrt,
                                          in_one_arg, in_temp, chinese_font_path, math_font_Path)

                    mask = parse_elements(label, sub_svg_node[:-2],
                                          offset_x,
                                          offset_y, scale,
                                          scaling_factor, h_offset,
                                          path_id_pair,
                                          mask, in_sub,
                                          in_frac, in_sqrt,
                                          in_one_arg, in_temp,
                                          chinese_font_path, math_font_Path)
                else:
                    mask = parse_elements(label, sub_svg_node[:-1],
                                          offset_x,
                                          offset_y, scale,
                                          scaling_factor, h_offset,
                                          path_id_pair,
                                          mask, in_sub,
                                          in_frac, in_sqrt,
                                          in_one_arg, in_temp,
                                          chinese_font_path, math_font_Path)

                in_sqrt -= 1
                rect_element = svg_node.find_all('rect')[-1]
                mask = get_rect_info(rect_element, offset_x, offset_y, h_offset, scale,
                                     scaling_factor, mask)
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x -= offset_x_sub * scale * scaling_factor
                        offset_y -= offset_y_sub * scale * scaling_factor
                ind += len(svg_node.find_all('g'))

            elif attr['data-mml-node'] == 'msub' or attr['data-mml-node'] == 'msup' or attr[
                'data-mml-node'] == 'msubsup':
                # pdb.set_trace()
                element = node.scription_node()
                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                        offset_x += offset_x_sub * scale * scaling_factor
                        offset_y += offset_y_sub * scale * scaling_factor
                        scale *= scale_sub
                in_sub += 1
                sub_svg_node = svg_node.find_all('g')
                mask = parse_elements(label, [sub_svg_node[0]],
                                      offset_x,
                                      offset_y,
                                      scale,
                                      scaling_factor, h_offset,
                                      path_id_pair, mask,
                                      in_sub,
                                      in_frac, in_sqrt,
                                      in_one_arg, in_temp,
                                      chinese_font_path, math_font_Path)

                if attr['data-mml-node'] == 'msubsup':
                    sup_ele = len(sub_svg_node[1].find_all('g'))
                    mask = parse_elements(label,
                                          sub_svg_node[1:sup_ele + 1],
                                          offset_x,
                                          offset_y,
                                          scale,
                                          scaling_factor, h_offset,
                                          path_id_pair, mask,
                                          in_sub,
                                          in_frac, in_sqrt,
                                          in_one_arg, in_temp,
                                          chinese_font_path, math_font_Path)

                    mask = parse_elements(label,
                                          sub_svg_node[sup_ele + 2:],
                                          offset_x,
                                          offset_y,
                                          scale,
                                          scaling_factor, h_offset,
                                          path_id_pair, mask,
                                          in_sub,
                                          in_frac, in_sqrt,
                                          in_one_arg, in_temp,
                                          chinese_font_path, math_font_Path)

                else:
                    mask = parse_elements(label, sub_svg_node[1:],
                                          offset_x,
                                          offset_y,
                                          scale,
                                          scaling_factor, h_offset,
                                          path_id_pair, mask,
                                          in_sub,
                                          in_frac, in_sqrt,
                                          in_one_arg, in_temp,
                                          chinese_font_path, math_font_Path)

                if 'transform' in attr:
                    if 'translate' in attr['transform']:
                        offset_x -= offset_x_sub * scale * scaling_factor
                        offset_y -= offset_y_sub * scale * scaling_factor
                in_sub -= 1
                ind += len(svg_node.find_all('g'))
        else:
            if 'transform' in attr:
                if 'translate' in attr['transform']:
                    offset_x_sub, offset_y_sub, scale_sub = get_offsets(attr)
                    offset_x = offset_x_sub * scaling_factor * scale + offset_x
                    offset_y = offset_y_sub * scaling_factor * scale + offset_y
                    scale *= scale_sub
        ind += 1
    return mask


def extract_trajectories(label, svg2xml, scaling_factor, path, config):
    '''
    Code to extract trajectories and transformation for all the elements
    in the svg
    :param label: str
        latex label
    :param svg2xml: xml tree
        svg converted to xml tree
    :param scaling_factor: float
        scaling factor to be used to scale the h and w
    :return: information's extracted from svg to inference the image
    '''
    chinese_font_path = config.CHINESE_FONT_PATH
    math_font_Path = config.MATH_FONT_PATH
    offset_x = 0.0
    offset_y = 0.0
    scale = 1
    in_sub = 0
    in_frac = 0
    in_sqrt = 0
    in_one_arg = 0
    in_temp = 0
    for element in svg2xml.find_all("svg"):
        attr = element.attrs
        if 'viewbox' in attr:
            box = attr['viewbox'].split()
            w_offset, h_offset, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            h_offset = h_offset * scaling_factor
    mask = np.ones((int(h * scaling_factor), int(w * scaling_factor))) * 255
    paths, attributes = svg2paths(path)
    path_id_pair = {}
    for i in range(len(attributes)):
        path_id_pair[attributes[i]['id']] = paths[i]
    label = [i for i in label.split() if i not in remove_elements]
    mask = parse_elements(label, svg2xml.find_all('g'), offset_x,
                          offset_y, scale,
                          scaling_factor,
                          h_offset,
                          path_id_pair, mask, in_sub, in_frac, in_sqrt,
                          in_one_arg,
                          in_temp,
                          chinese_font_path, math_font_Path)
    cv2.imwrite(join(config.OUT_SVG_IMAGE_PATH, '{}.png'.format(path.split('/')[-1].split('.')[0])), mask)
    return mask
