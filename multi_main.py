import json
import bs4
import glob
import time
from os.path import join
import random
from multiprocessing import Process
import multiprocessing
from project_config import ProjectConfig
from src.new_parse_svg import parse_svg
from src.generate_svg import generate_svg
from src.inference_image import inference_image
from src.load_meta_data import load_source_labels, get_drawable_math_symbols, \
    get_chinese_symbols, load_character_dict, load_crohme_dict, load_deletion_symbols
import tempfile
from src.util.file_util import load_json, check_out_dir, save_json
from src.util.util import preprocess_labels, postprocess_labels
from src.meta2virtual import meta2virtual
from src.deletion import add_deletion
from src.parse_label import parse_label
import traceback
import pdb


def generate_data(label, svg_path, offset_position, id, chinese_symbols, valid_math_symbols, path_list, author,
                  crohme_path_list, casia_character_dict, config,
                  incomplete_deletion_flag, augmented_samples):
    # print(label)
    generate_svg(label, svg_path)
    svg = open(svg_path, 'r').readlines()[0]

    # Parse svg
    try:
        parsed_label, offset_list = parse_svg(svg, svg_path, config, label)
        parsed_label = add_deletion(parsed_label, chinese_symbols, valid_math_symbols, path_list, author,
                                    config.IGNORE_CHAR_DELETION)
    except Exception as inst:
        print(inst)
        # traceback.print_tb(inst.__traceback__)
        print(label)
        return

    # Inference image given the virtual image
    try:
        image, label_inf = inference_image(parsed_label, path_list, crohme_path_list, delete_ims, config,
                                           incomplete_deletion_flag, id, ProjectConfig.OFFSET_POSITION, author,
                                           config.USE_LAYOUT_MODEL)
    except Exception as inst:
        # print(inst)
        # print(label)
        return
    h, w = image.shape
    augmented_samples[id] = {'h': h,
                             'w': w,
                             'label': postprocess_labels(label_inf),
                             'real_label': postprocess_labels(label)}


if __name__ == '__main__':
    config = ProjectConfig()
    casia_character_dict = load_json(config.CASIA_CHARACTER_DATA)
    path_list = load_character_dict(config)
    delete_ims = load_deletion_symbols(config)
    crohme_path_list = load_crohme_dict(config)
    chinese_symbols = get_chinese_symbols(path_list)
    valid_math_symbols = get_drawable_math_symbols(config)
    labels = load_source_labels(config, valid_math_symbols)
    check_out_dir(config.OUT_PATH)
    check_out_dir(config.OUT_SVG_IMAGE_PATH)
    check_out_dir(config.DELETE_OUT_PATH)
    check_out_dir(config.SVG_PATH)
    labels = load_json(config.LABELS_PATH)
    incomplete_deletion_flag = config.INCOMPLETE_DELETION
    one_arg_lalels_list = {}
    num_samples = len(labels)
    count = 0
    start = time.time()
    processes = []
    label_keys = list(labels.keys())
    manager = multiprocessing.Manager()
    augmented_samples = manager.dict()
    valid_math_symbols -= {'.', ',', ':', '"', '-', ';'}
    while count < num_samples:
        for j in range(0, min(num_samples - count, 10)):
            if count != 0 and count % 100 == 0:
                print('processed: {}/{}'.format(count, num_samples))
                print('time taken: {:.2f}'.format(time.time() - start))
                start = time.time()

            label = labels[label_keys[count]]['label']
            try:
                label = preprocess_labels(label, config)
            except Exception as inst:
                # pdb.set_trace()
                print(inst)
                continue
            if any(item in ProjectConfig.IGNORE_SAMPLES_LIST for item in label.split()):
                continue
            id = label_keys[count]
            svg_path = join(config.SVG_PATH, '{}.svg'.format(id))

            # Randomly select author
            author = random.choice(list(path_list.keys()))

            job = Process(target=generate_data, args=(
                label, svg_path, config.OFFSET_POSITION, id, chinese_symbols, valid_math_symbols, path_list, author,
                crohme_path_list, casia_character_dict, config, incomplete_deletion_flag, augmented_samples))
            processes.append(job)
            count += 1
        for p in processes:
            p.daemon = True
            p.start()
        for p in processes:
            p.join()
        processes = []
    save_json('augmented_samples.json', augmented_samples.copy())
