import math
import pdb

special_arguments = ['20D7', '5E', 'AF']


def initialize_positions():
    min_x = math.inf
    min_y = math.inf
    max_x = 0
    max_y = 0
    return min_x, min_y, max_x, max_y


def cal_box_size(virtual_image):
    '''
    Given the list of character informations calculate the
    overall size of the box

    :param virtual_image: List[Dict]
    :return: min and max of x and y coordinates
    '''
    min_x, min_y, max_x, max_y = initialize_positions()
    for vim in virtual_image:
        sample = vim
        lu_position = sample['lu_position']
        min_x = min(min_x, lu_position[0])
        min_y = min(min_y, lu_position[1])
        max_x = max(max_x, lu_position[0] + abs(sample['w']))
        max_y = max(max_y, lu_position[1] + abs(sample['h']))
    return min_x, min_y, max_x, max_y


def get_offset(virtual_image, prev_coord, new_vim, prev_coord_list, offset_position='up'):
    '''
    recursively finds the position offset from previous character

    :param virtual_image: List[Dict]
        list of all characters sorted based on the character position
    :param prev_coord: [int, int]
        right up or middle right position of previous character
    :param new_vim: List[Dict]
        new virtual image with offset
    :param prev_coord_list: List[[int, int]]
        keeping track of all previous coordinates
    :param offset_position: str
        specifying the offset position (up or middle)
    :return: List[Dict]
        new virtual image with offset
    '''
    ind = 0
    while ind < len(virtual_image):
        if virtual_image[ind]['type'] != 'frac_rect' and virtual_image[ind]['type'] != 'root_rect' and \
                virtual_image[ind]['id'].split('-')[-1] not in special_arguments:
            if offset_position == 'up':
                virtual_image[ind]['offset'] = [virtual_image[ind]['lu_position'][0] - prev_coord[0],
                                                virtual_image[ind]['lu_position'][1] - prev_coord[1]]
            elif offset_position == 'middle':
                virtual_image[ind]['offset'] = [virtual_image[ind]['lu_position'][0] - prev_coord[0],
                                                (virtual_image[ind]['lu_position'][1] + virtual_image[ind]['h'] // 2) -
                                                prev_coord[1]]
            new_vim.append(virtual_image[ind])
            if offset_position == 'up':
                prev_coord = [virtual_image[ind]['lu_position'][0] + virtual_image[ind]['w'],
                              virtual_image[ind]['lu_position'][1]]
            elif offset_position == 'middle':
                prev_coord = [virtual_image[ind]['lu_position'][0] + virtual_image[ind]['w'],
                              virtual_image[ind]['lu_position'][1] + virtual_image[ind]['h'] // 2]
            prev_coord_list.append(prev_coord)
            ind += 1

        elif virtual_image[ind]['type'] == 'frac_rect':
            rect = virtual_image[ind]
            rect_ind = ind
            x_range = list(range(rect['lu_position'][0], rect['lu_position'][0] + abs(rect['w'])))
            num_elements = []
            den_elements = []
            min_x = rect['lu_position'][0]
            min_y = math.inf
            max_x = rect['lu_position'][0] + abs(rect['w'])
            max_y = 0
            temp_ind = 0
            while temp_ind < len(virtual_image[ind + 1:]):
                sample = virtual_image[ind + temp_ind + 1]
                lu_position = sample['lu_position']
                if lu_position[0] in x_range:
                    min_x = min(min_x, lu_position[0])
                    min_y = min(min_y, lu_position[1])
                    max_x = max(max_x, lu_position[0] + abs(sample['w']))
                    max_y = max(max_y, lu_position[1] + abs(sample['h']))
                    if lu_position[1] < rect['lu_position'][1]:
                        num_elements.append(sample)
                    else:
                        den_elements.append(sample)
                temp_ind += 1
            rect['min_x'] = min_x
            rect['min_y'] = min_y
            rect['max_x'] = max_x
            rect['max_y'] = max_y
            rect['global_offset'] = [min_x - prev_coord[0], min_y - prev_coord[1]]
            prev_coord = [min_x, min_y]
            prev_coord_list.append(prev_coord)
            get_offset(num_elements, prev_coord, new_vim, prev_coord_list, offset_position)
            min_x, min_y, max_x, max_y = cal_box_size(new_vim[len(new_vim) - len(num_elements):])
            rect['num_offset'] = [rect['lu_position'], [min_x, max_y]]
            ind += len(num_elements)
            prev_coord = [rect['lu_position'][0], rect['lu_position'][1] + rect['h']]
            prev_coord_list.append(prev_coord)
            get_offset(den_elements, prev_coord, new_vim, prev_coord_list, offset_position)
            min_x, min_y, max_x, max_y = cal_box_size(new_vim[len(new_vim) - len(den_elements):])
            rect['den_offset'] = [[rect['min_x'], rect['lu_position'][1] + rect['h']], [min_x, min_y]]
            ind += len(den_elements)
            new_vim.insert(len(new_vim) - len(num_elements) - 1, rect)
            ind += 1
            prev_coord = [rect['max_x'], rect['min_y']]
            prev_coord_list.append(prev_coord)

        elif virtual_image[ind]['type'] == 'root_rect':
            rect = virtual_image[ind]
            rect_ind = ind
            x_range = list(range(rect['lu_position'][0], rect['lu_position'][0] + abs(rect['w'])))
            elements = []
            temp_ind = 0
            while temp_ind < len(virtual_image[ind + 1:]):
                sample = virtual_image[ind + temp_ind + 1]
                lu_position = sample['lu_position']
                if lu_position[0] in x_range:
                    elements.append(sample)
                temp_ind += 1
            if virtual_image[ind + len(elements) + 1]['id'].split('-')[-1] == '221A':
                elements.insert(0, virtual_image[ind + len(elements) + 1])
            else:
                virtual_image[ind + len(elements) + 1]['offset'] = [
                    virtual_image[ind + len(elements) + 1]['lu_position'][0] - prev_coord[0],
                    virtual_image[ind + len(elements) + 1]['lu_position'][1] - prev_coord[1]]
                new_vim.append(virtual_image[ind + len(elements) + 1])
                elements.insert(0, virtual_image[ind + len(elements) + 2])
                ind += 1

            get_offset(elements, prev_coord, new_vim, prev_coord_list, offset_position)
            rect['offset'] = [0, 0]
            rect['min_x'] = elements[0]['lu_position'][0]
            rect['max_y'] = elements[0]['lu_position'][1] + elements[0]['h']
            new_vim.insert(len(new_vim) - len(elements), rect)
            ind += 1
            ind += len(elements)
            prev_coord = [rect['lu_position'][0] + rect['w'], rect['lu_position'][1]]
            prev_coord_list.append(prev_coord)

        elif virtual_image[ind]['id'].split('-')[-1] in special_arguments:
            spec_element = virtual_image[ind]
            x_range = list(
                range(spec_element['lu_position'][0], spec_element['lu_position'][0] + abs(spec_element['w'])))
            elements = [virtual_image[ind - 1]]
            temp_ind = 0
            while temp_ind < len(new_vim[:ind - 1]):
                sample = virtual_image[ind - temp_ind - 2]
                lu_position = sample['lu_position']
                if lu_position[0] in x_range or lu_position[0] + sample['w'] in x_range or (
                        lu_position[0] + sample['w'] - lu_position[0]) // 2 + lu_position[0] in x_range:
                    if spec_element['lu_position'][1] < lu_position[1] < spec_element['lu_position'][1] + 60:
                        elements.append(sample)
                temp_ind += 1
            elements = elements[::-1]
            del new_vim[len(new_vim) - len(elements):]
            ind -= len(elements)
            elements.append(spec_element)
            min_x, min_y, max_x, max_y = cal_box_size(elements)
            elements = elements[:-1]
            prev_coord = prev_coord_list[len(prev_coord_list) - len(elements) - 1]
            spec_element['global_offset'] = [min_x - prev_coord[0], min_y - prev_coord[1]]
            spec_element['offset'] = [spec_element['lu_position'][0] - min_x, spec_element['lu_position'][1] - min_y]
            prev_coord = [min_x, min_y]
            prev_coord_list.append(prev_coord)
            get_offset(elements, prev_coord, new_vim, prev_coord_list, offset_position)
            prev_coord = [max_x, min_y]
            prev_coord_list.append(prev_coord)
            ind += len(elements)
            min_x, min_y, max_x, max_y = cal_box_size(elements)
            spec_element['left_offset'] = [[min_x, min_y], [spec_element['lu_position'][0],
                                                            spec_element['lu_position'][1] + spec_element['h']]]
            spec_element['right_offset'] = [[max_x, min_y], [spec_element['lu_position'][0] + spec_element['h'],
                                                             spec_element['lu_position'][1] + spec_element['h']]]
            new_vim.append(spec_element)
            ind += 1

    return new_vim


def meta2virtual(svg_image, label, id, offset_position):
    '''
    adding offset element to the characters
    :param svg_image: Dict[Dict[List]]
        keys: characters, texts, root_rect, frac_rect, canvas_h, canvas_w, img_id
        virtual image that has all the information's to inference the image
    :param label: str
    :param id: str
        image id
    :param offset_position: str
        specifying the offset position (up or middle)
    :return: List[Dict]
        new virtual image
    '''
    virtual_image = []
    virtual_char = svg_image['characters']
    virtual_texts = svg_image['texts']
    virtual_root_rect = svg_image['root_rect']
    virtual_frac_rect = svg_image['frac_rect']

    if virtual_char != []:
        for char in virtual_char:
            virtual_image.append(char)

    if virtual_texts != []:
        for text in virtual_texts:
            virtual_image.append(text)

    if virtual_root_rect != []:
        for root_rect in virtual_root_rect:
            virtual_image.append(root_rect)

    if virtual_frac_rect != []:
        for frac_rect in virtual_frac_rect:
            virtual_image.append(frac_rect)

    # Sort the symbols based on the position of the character
    virtual_image = sorted(virtual_image, key=lambda d: (d['pos']))
    virtual_image = get_offset(virtual_image, [0, 0], [], [[0, 0]], offset_position)
    # Sort the symbols based on the pos of the character again to make sure its aligned
    virtual_image = sorted(virtual_image, key=lambda d: (d['pos']))
    virtual_image.append({'type': 'canvas', 'img_id': svg_image['img_id'],
                          'canvas_h': svg_image['canvas_h'], 'canvas_w': svg_image['canvas_w']})
    return virtual_image
