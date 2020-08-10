import json
import glob
import traceback
from os.path import join
import random
from project_config import ProjectConfig
from src.new_parse_svg import parse_svg
from src.generate_svg import generate_svg
from src.inference_image import inference_image
from src.load_meta_data import load_source_labels, get_drawable_math_symbols, \
    get_chinese_symbols, load_character_dict, load_crohme_dict, load_deletion_symbols, load_new_deletion_symbol
import tempfile
from src.util.file_util import load_json, check_out_dir, save_json
from src.util.util import preprocess_labels, postprocess_labels
from src.meta2virtual import meta2virtual
from src.deletion import add_deletion
from src.parse_label import parse_label

import matplotlib.pyplot as plt
import pdb

if __name__=='__main__':
    config = ProjectConfig()
    # casia_character_dict = load_json(config.CASIA_CHARACTER_DATA)
    path_list = load_character_dict(config)
    delete_ims = load_new_deletion_symbol(config)
    crohme_path_list = load_crohme_dict(config)
    chinese_symbols = get_chinese_symbols(path_list)
    valid_math_symbols = get_drawable_math_symbols(config)
    labels = load_source_labels(config, valid_math_symbols)
    check_out_dir(config.OUT_SVG_IMAGE_PATH)
    labels = load_json(config.LABELS_PATH)
    incomplete_deletion_flag = config.INCOMPLETE_DELETION
    one_arg_lalels_list = {}
    augmented_samples = {}
    # pdb.set_trace()
    valid_math_symbols -= {'.', ',', ':', '"', '-', ';', '|'}
    count = 0
    total_offset_list = []
    for key, val in labels.items():
        with tempfile.TemporaryDirectory() as tmp_svg_path:
            label = val['label']
            # label = ' 故 = \\frac { S _ { 小 圆 } } { S _ { 大 圆 } } = \\frac { \\pi \\( \\frac { 1 } { 2 } \\) ^ { 2 } } { \\pi 1 ^ { 2 } } = \\frac { 1 } { 4 } '
            # label =  ' \\therefore P = \\frac { S 阴 影 } { S 矩 形 } = \\frac { \\frac { 2 } { 3 } } { 2 } = \\frac { 1 } { 3 } '
            # label = '\\cdots'
            label = preprocess_labels(label, config)
            if any(item in ProjectConfig.IGNORE_SAMPLES_LIST for item in label.split()):
                continue
            id = key
            print(id)
            print(label)
            svg_path = join(tmp_svg_path, '{}.svg'.format(id))
            
            # Randomly select author
            author = random.choice(list(path_list.keys()))

            # Generate svg for the labels using mathjax 
            generate_svg(label, svg_path)
            svg = open(svg_path,'r').readlines()[0]
            # print(svg)
            # Parse svg
            try:
                parsed_label = parse_svg(svg, svg_path, config, label)
            except Exception as e:
                print(e)
                continue