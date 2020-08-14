import json
import glob
import traceback
from os.path import join
import random
from project_config import ProjectConfig
from src.new_parse_svg import parse_svg
from src.generate_svg import generate_svg
import tempfile
from src.util.file_util import load_json, check_out_dir, save_json
from src.util.util import preprocess_labels, postprocess_labels

import matplotlib.pyplot as plt
import pdb

if __name__=='__main__':
    config = ProjectConfig()
    check_out_dir(config.OUT_SVG_IMAGE_PATH)
    labels = load_json(config.LABELS_PATH)
    incomplete_deletion_flag = config.INCOMPLETE_DELETION
    one_arg_lalels_list = {}
    augmented_samples = {}
    count = 0
    total_offset_list = []
    for key, val in labels.items():
        with tempfile.TemporaryDirectory() as tmp_svg_path:
            label = val['label']
            # label = ' 故 = \\frac { S _ { 小 圆 } } { S _ { 大 圆 } } = \\frac { \\pi \\( \\frac { 1 } { 2 } \\) ^ { 2 } } { \\pi 1 ^ { 2 } } = \\frac { 1 } { 4 } '
            # label =  ' \\therefore P = \\frac { S 阴 影 } { S 矩 形 } = \\frac { \\frac { 2 } { 3 } } { 2 } = \\frac { 1 } { 3 } '
            # label = ' 解 : 直 线 \\rho c o s ( \\theta - \\frac { \pi } { 3 } ) = \\frac { 1 } { 2 } , 即 \\frac { 1 } { 2 } \\rho c o s \\theta + \\frac { \sqrt { 3 } } { 2 } \\rho s i n \\theta - \\frac { 1 } { 2 } = 0 的 直 角 坐 标 方 程 为 \\frac { 1 } { 2 } x + \\frac { \sqrt { 3 } } { 2 } y - \\frac { 1 } { 2 } = 0 , '
            label = preprocess_labels(label, config)
            if any(item in ProjectConfig.IGNORE_SAMPLES_LIST for item in label.split()):
                continue
            id = key
            print(id)
            print(label)
            svg_path = join(tmp_svg_path, '{}.svg'.format(id))

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
            # pdb.set_trace()