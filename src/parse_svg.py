import bs4
import re
import cv2
from os.path import join
import json
import numpy as np
from svgpathtools import svg2paths
from svg.path import parse_path
from PIL import ImageFont, ImageDraw, Image
import pdb

 
r1 = re.compile(".*?\((.*?)\)") 
    
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
    transform_x_all, transform_y_all, scale_all, id, mask, h_offset, texts, text_transform_x_all, text_transform_y_all,\
                            root_rect, frac_rect, shift_char, text_pos, char_pos, frac_rect_pos, root_rect_pos = extract_trajectories(label, svg2xml, scaling_factor)
                            
    virtual_image = get_virtual_image(path, transform_x_all, transform_y_all, scale_all, id, mask, h_offset, texts, 
                text_transform_x_all, text_transform_y_all, root_rect, frac_rect, shift_char,  text_pos, char_pos, frac_rect_pos, root_rect_pos, scaling_factor, config)
    return virtual_image
    
    
def get_virtual_image(path, transform_x_all, transform_y_all, scale_all, id, mask, h_offset, texts, text_transform_x_all, 
                        text_transform_y_all, root_rect, frac_rect, shift_char, text_pos, char_pos, frac_rect_pos, root_rect_pos, scaling_factor, config):
    '''
    Code to get virtual image from the extracted 
    trajectories. This code also prints out the virtual 
    image for dubuging.
    '''
    virtual_image = {}
    virtual_image['characters'] = []
    virtual_image['texts'] = []
    virtual_image['root_rect'] = []
    virtual_image['frac_rect'] = []
    virtual_image['canvas_h'] = mask.shape[0]      
    virtual_image['canvas_w'] = mask.shape[1]  
    virtual_image['img_id'] = path.split('/')[-1].split('.')[0] 
    paths, attributes = svg2paths(path)
    new_dict = {}
    for i in range(len(attributes)):
        new_dict[attributes[i]['id']] = paths[i]

    for i in range(len(id)):
        # pdb.set_trace()
        scale = scale_all[i]
        path_string = new_dict[id[i]]
        x = transform_x_all[i]
        y = transform_y_all[i]
        contour = []
        for e in path_string: 
            contour.append([[e.start.real, e.start.imag]])
            contour.append([[e.end.real, e.end.imag]]) 
        contour = np.array(contour) * scale * scaling_factor
        try:
            contour[:,:,0] = contour[:,:,0]*1 + contour[:,:,1]*0 + 0 + x
        except:
            continue
        contour[:,:,1] = contour[:,:,0]*0 + contour[:,:,1]*(-1) + 0 - h_offset - y 
        contour = contour.astype(int)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        min_x, max_x = contour[:,:,0].min(), contour[:,:,0].max()
        min_y, max_y = contour[:,:,1].min(), contour[:,:,1].max()
        cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, 2) 
        virtual_image['characters'].append({'type':'character', 'id':id[i], 'lu_position': [min_x, min_y], 'h': max_y-min_y, 'w': max_x-min_x, 'pos': char_pos[i], 'scale': scale})
        
    if root_rect != []:
        for ind, box in enumerate(root_rect):
            box[1] = abs(box[1] + h_offset)
            box[3] = abs(box[3] + h_offset)
            box = [int(i) for i in box]
            cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), 255, -1)
            virtual_image['root_rect'].append({'id':'-', 'lu_position': [box[0], box[1]], 'type': 'root_rect',
                                                'h': box[3]-box[1], 'w': box[2]-box[0], 'pos': root_rect_pos[ind]})

    if frac_rect != []:
        for ind, box in enumerate(frac_rect):
            box[1] = abs(box[1] + h_offset)
            box[3] = abs(box[3] + h_offset)
            box = [int(i) for i in box]
            pdb.set_trace()
            cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), 255, -1)
            virtual_image['frac_rect'].append({'id':'-', 'lu_position': [box[0], box[1]], 'type': 'frac_rect', 
                                                'h': box[3]-box[1], 'w': box[2]-box[0], 'pos': frac_rect_pos[ind]})
                                                
    if texts != []:
        img_pil = Image.fromarray(mask.astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)
        for i in range(len(texts)):
            font_size = int(884*scaling_factor)
            fontpath = config.CHINESE_FONT_PATH 
            font = ImageFont.truetype(fontpath, font_size)
            text_transform_x_all[i], text_transform_y_all[i] = int(text_transform_x_all[i]), int(text_transform_y_all[i])
            # print(text_transform_x_all, text_transform_y_all)
            if shift_char:            
                text_transform_y_all[i] = int(text_transform_y_all[i] - h_offset/2)
            draw.text((text_transform_x_all[i], text_transform_y_all[i]), texts[i], font = font, fill = 255)           
            draw.rectangle([(text_transform_x_all[i], text_transform_y_all[i]), 
                                    (text_transform_x_all[i]+font_size, text_transform_y_all[i]+font_size)], None, 255)
            virtual_image['texts'].append({'type':'text', 'id':texts[i], 'lu_position': [text_transform_x_all[i], text_transform_y_all[i]],
                                    'h': font_size, 'w': font_size, 'pos': text_pos[i]})
        # img_pil.save(join(config.OUT_SVG_IMAGE_PATH,'{}.png'.format(path.split('/')[-1].split('.')[0])))
        # pdb.set_trace()
        mask = np.array(img_pil)
    cv2.imwrite(join(config.OUT_SVG_IMAGE_PATH,'{}.png'.format(path.split('/')[-1].split('.')[0])), mask)
    return virtual_image


def extract_trajectories(label, svg2xml, scaling_factor):
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
    id_all = []
    scale_all = []
    transform_x = 0
    transform_y = 0
    scale_m1 = 1
    offset_x = 0.0
    offset_y = 0.0
    transform_x_all = []
    transform_y_all = []
    text_transform_x_all = []
    text_transform_y_all = []
    root_rect = []
    frac_rect = []
    texts = []
    drop = False
    shift_char = False
    temp_x = []
    temp_y = []
    temp_scale = []
    count = []
    scale_count = []
    char_pos = []
    text_pos = []
    root_rect_pos = []
    frac_rect_pos = []
    pos = 0
    unsupported_characetrs = ['\\RomanNumeralCaps', '\\textperthousand', '\\romannumeral', '\\RomanNumeralCaps', '\\textcircled']
    # if '\\vec' in label:
        # pdb.set_trace()
    # pdb.set_trace()
    for element in svg2xml.find_all("g"):
        attr = element.attrs
        if 'fill' in attr:
            if attr['fill'] == 'red':
                # drop the sample
                # print('invalid svg')
                if not any(item in unsupported_characetrs for item in label.split()):
                    print(label)
                raise Exception('INVALID SVG')
                               
        if 'transform' in attr:
            if 'translate' in attr['transform']:
                offsets = r1.match(attr['transform'])
                m1 = offsets.group(1)   

                if len(element.find_all('g'))>0:
                    offset_x += float(m1.split(',')[0]) * scale_m1 
                    offset_y += float(m1.split(',')[1]) * scale_m1 
                    count.append(len(element.find_all('g')) + 1)
                    temp_x.append(float(m1.split(',')[0]) * scale_m1 )
                    temp_y.append(float(m1.split(',')[1]) * scale_m1 )
                    transform_x = offset_x                    
                    transform_y = offset_y 
                else:                       
                    transform_x = offset_x + float(m1.split(',')[0]) * scale_m1 
                    transform_y = offset_y + float(m1.split(',')[1]) * scale_m1 
                if 'scale' in attr['transform']:
                    scale = attr['transform'].split()[-1]
                    scale_off = r1.match(scale)
                    scale_temp = float(scale_off.group(1))
                    scale_m1 *= scale_temp
                    scale_count.append(len(element.find_all('g')) + 1)
                    temp_scale.append(float(scale_off.group(1)))
        
        # Get all the rect boxes for fraction and square root
        if 'data-mml-node' in attr:
            if attr['data-mml-node'] == 'merror': 
                raise Exception('delimeter error')
            if attr['data-mml-node'] == 'mfrac':        
                shift_char = True
            if attr['data-mml-node'] == 'msqrt' or attr['data-mml-node'] == 'mfrac' or attr['data-mml-node'] == 'mroot':
                rect_element = element.find_all('rect')[-1]
                if attr['data-mml-node'] == 'mfrac':
                    frac_rect.append([float(rect_element['x'])*scale_m1 + offset_x, float(rect_element['y'])*scale_m1 + offset_y, 
                                float(rect_element['x'])*scale_m1 + float(rect_element['width'])*scale_m1 + offset_x, 
                                float(rect_element['y'])*scale_m1 - float(rect_element['height'])*scale_m1 + offset_y])                
                    # pdb.set_trace()
                    if pos in char_pos:
                        pos+=1
                    frac_rect_pos.append(pos)
                    pos+=1
                    
                elif attr['data-mml-node'] == 'msqrt' or attr['data-mml-node'] == 'mroot':
                    # print(scale_m1)
                    root_rect.append([float(rect_element['x'])*scale_m1 + offset_x, float(rect_element['y'])*scale_m1 + offset_y, 
                                float(rect_element['x'])*scale_m1  + float(rect_element['width'])*scale_m1 + offset_x, 
                                float(rect_element['y'])*scale_m1  - float(rect_element['height'])*scale_m1 + offset_y]) 
                    if pos in char_pos:
                        pos+=1
                    root_rect_pos.append(pos)
                    pos+=1

        # Get all the tranformations for unicoded characters
        if len(element.find_all('g'))==0 and not element.find_all('text'):
            for ele in element.find_all('use'):
                temp_char_x = 0
                temp_char_y = 0
                # if len(element.find_all('use')) > 1:
                    # pdb.set_trace()
                if 'transform' in ele.attrs:
                    offsets = r1.match(ele.attrs['transform'])
                    m1 = offsets.group(1)  
                    temp_char_x = float(m1.split(',')[0])
                    temp_char_y = float(m1.split(',')[1])
                id = ele.attrs['xlink:href'].replace('#','')
                id_all.append(id)
                transform_x_all.append(transform_x+temp_char_x)
                transform_y_all.append(transform_y+temp_char_y)
                scale_all.append(scale_m1)
                if pos in char_pos:
                    pos+=1
                char_pos.append(pos)
                pos+=1
            transform_x = 0
            transform_y = 0
        else:
                                                                                                                                                                                                                                        
            offset_x_temp = 0
            offset_y_temp = 0
            # Get all the texts 
            if len(element.find_all('g')) == 0:
                # pdb.set_trace() 
                if 'transform' in attr:
                    offsets = r1.match(attr['transform'])
                    m1 = offsets.group(1) 
                    offset_x_temp = float(m1.split(',')[0])
                    offset_y_temp = float(m1.split(',')[1]) 
                if element.use is not None: 
                    for i, temp in enumerate(element):
                        if 'xlink:href' in temp.attrs:
                            # pdb.set_trace()
                            if 'transform' in temp.attrs:
                                offsets = r1.match(temp.attrs['transform'])
                                m1 = offsets.group(1) 
                                id = temp.attrs['xlink:href'].replace('#', '')
                                id_all.append(id)
                                transform_x_all.append(offset_x_temp+float(m1.split(',')[0])+offset_x)
                                transform_y_all.append(offset_y_temp+float(m1.split(',')[1])+offset_y)
                                scale_all.append(scale_m1)
                                char_pos.append(pos+i)
                                # print(temp)
                            else:
                                id = temp.attrs['xlink:href'].replace('#','')
                                id_all.append(id)
                                transform_x_all.append(offset_x+offset_x_temp)
                                transform_y_all.append(offset_y+offset_y_temp)
                                scale_all.append(scale_m1)
                                char_pos.append(pos+i)     
                                
                # pdb.set_trace()
                for tag in element.find_all('text'):
                    if 'translate' in tag['transform']:
                        translate = tag['transform'].split('m')[0]
                        val = r1.match(translate).group(1)
                        texts.append(tag.string)
                        text_transform_x_all.append(float(val.split(',')[0]) + offset_x_temp + offset_x)
                        text_transform_y_all.append(float(val.split(',')[1]) + offset_y_temp + offset_y)
                        if pos in char_pos or pos in text_pos:
                            pos+=1
                        text_pos.append(pos)
                        pos+=1
                    else:
                        texts.append(tag.string)
                        text_transform_x_all.append(offset_x + offset_x_temp)
                        text_transform_y_all.append(offset_y + offset_y_temp)                        
                        if pos in char_pos or pos in text_pos:
                            pos+=1
                        text_pos.append(pos)
                        pos+=1
                        
        # To keep track of offset transformation
        for c in range(len(count)):
            if count[c] != 0:
                count[c]-=1

            if count[c] == 0:
                offset_x -= temp_x[c]
                offset_y -= temp_y[c]
                temp_x[c] = 0
                temp_y[c] = 0
        
        # TO keep track of scaling 
        for s in range(len(scale_count)):
            if scale_count[s] != 0:
                scale_count[s]-=1

            if scale_count[s] == 0:
                scale_m1 /= temp_scale[s]
                temp_scale[s] = 1
    
    # Get the canvas information
    for element in svg2xml.find_all("svg"):
        attr = element.attrs
        if 'viewbox' in attr:
            box = attr['viewbox'].split()
            w_offset, h_offset, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if 'height' in attr:
            h_scale = float(attr['height'].replace('ex',''))
        if 'width' in attr:
            w_scale = float(attr['width'].replace('ex',''))

    # scale all the parameters based on the scaling factor
    mask = np.zeros((int(h*scaling_factor),int(w*scaling_factor)))
    # print(mask.shape)
    if transform_x_all!=[]:
        transform_x_all = [i*scaling_factor for i in transform_x_all]
        transform_y_all = [i*scaling_factor for i in transform_y_all]
    if text_transform_x_all!=[]:
        text_transform_x_all = [i*scaling_factor for i in text_transform_x_all]
        text_transform_y_all = [i*scaling_factor for i in text_transform_y_all]
    if root_rect!=[]:
        new_rect = []
        for sample in root_rect:
            sample = [i*scaling_factor for i in sample]
            new_rect.append(sample)
        root_rect = new_rect
    if frac_rect!=[]:
        new_rect = []
        for sample in frac_rect:
            sample = [i*scaling_factor for i in sample]
            new_rect.append(sample)
        frac_rect = new_rect
    return transform_x_all, transform_y_all, scale_all, id_all, mask, h_offset*scaling_factor, texts, text_transform_x_all, \
            text_transform_y_all, root_rect, frac_rect, shift_char, text_pos, char_pos, frac_rect_pos, root_rect_pos