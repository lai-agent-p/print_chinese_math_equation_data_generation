import numpy as np
import cv2
import torch

from src.util.nn_util import save_network, load_last_checkpoint, load_best_checkpoint
import numpy as np
from src.util.image_util import otsu_thresh, pad2square
from src.model import train_step, LayoutModel
from src.layoutlearning_config import ProjectConfig
from os.path import join


def form_im(start_lu_pos, h_centers_diff, gap_diffs, line_slope, base_gap, ims, max_h, max_w):
    # convert to virtual im
    im_boxes = []
    first_im = ims[0]
    h, w = first_im.shape
    h_up = h // 2
    h_down = h - h_up
    cur_center, cur_x = 0 + int(h_centers_diff[0]), 0
    im_boxes.append([[cur_center - h_up, 0], [0 + h_down, w]])
    for i in range(1, len(ims)):
        h, w = ims[i].shape
        h_up = h // 2
        h_down = h - h_up
        prev_x = im_boxes[-1][1][1]
        cur_x = prev_x + int(gap_diffs[i - 1] + base_gap)
        cur_center = int(cur_x * line_slope + h_centers_diff[i])
        im_boxes.append([[cur_center - h_up, cur_x], [cur_center + h_down, cur_x + w]])
    # convert h
    min_h = 0
    min_w = 0
    for box in im_boxes:
        min_h = min(box[0][0], min_h)
        min_w = min(box[0][1], min_w)
    for i, box in enumerate(im_boxes):
        box[0][0] -= min_h
        box[1][0] -= min_h
        box[0][1] -= min_w
        box[1][1] -= min_w

        # if i == 0:
        #     box[0][0] = 0
        #     box[1][0] = 0
        #     box[0][1] = 0
        #     box[1][1] = 0

        box[0][0] += start_lu_pos[1]
        box[1][0] += start_lu_pos[1]
        box[0][1] += start_lu_pos[0]
        box[1][1] += start_lu_pos[0]

    im_boxes = np.asarray(im_boxes)
    lu_poses = [im_box[0] for im_box in im_boxes]
    lu_poses = [[lu_pos[1], lu_pos[0]] for lu_pos in lu_poses]
    last_h, last_w = ims[-1].shape
    next_pos = [[lu_pos[0] + last_w, lu_pos[1] + last_h // 2] for lu_pos in lu_poses]
    return lu_poses, next_pos


def get_vis_im(h_centers_diff, gap_diffs, line_slope, base_gap, ims, max_h, max_w, content):
    im_boxes = []
    first_im = ims[0]
    h, w = first_im.shape
    h_up = h // 2
    h_down = h - h_up
    cur_center, cur_x = 0 + int(h_centers_diff[0]), 0
    im_boxes.append([[cur_center - h_up, 0], [0 + h_down, w]])
    for i in range(1, len(ims)):
        h, w = ims[i].shape
        h_up = h // 2
        h_down = h - h_up
        prev_x = im_boxes[-1][1][1]
        cur_x = prev_x + int(gap_diffs[i - 1] + base_gap)
        cur_center = int(cur_x * line_slope + h_centers_diff[i])
        im_boxes.append([[cur_center - h_up, cur_x], [cur_center + h_down, cur_x + w]])
    # convert h
    min_h = 0
    min_w = 0
    for box in im_boxes:
        min_h = min(box[0][0], min_h)
        min_w = min(box[0][1], min_w)
    for box in im_boxes:
        box[0][0] -= min_h
        box[1][0] -= min_h
        box[0][1] -= min_w
        box[1][1] -= min_w

    if min_w < 0:
        a = 3
    im_boxes = np.asarray(im_boxes)
    canvas_shape = [np.max(im_boxes[:, 1, 0]), im_boxes[-1, 1, 1]]
    canvas_h, canvas_w = canvas_shape
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    start_point = (0, -min_h)
    end_point = (canvas_w, max(min(int(line_slope * canvas_w - min_h), canvas_shape[1]), 0))
    canvas = cv2.line(canvas, start_point, end_point, color=(0, 0, 255), thickness=2)
    for i, im in enumerate(ims):
        box = im_boxes[i]
        im_shape = [box[1, 0] - box[0, 0], box[1, 1] - box[0, 1]]
        im = cv2.resize(im, (im_shape[1], im_shape[0]))
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        canvas = cv2.rectangle(canvas, (box[0, 1], box[0, 0]), (box[1, 1], box[1, 0]), color=(255, 0, 0), thickness=2)
        canvas = cv2.circle(canvas, (box[0, 1], int((box[0, 0] + box[1, 0]) // 2)), radius=2, color=(0, 255, 0),
                            thickness=-1)
        canvas = cv2.putText(canvas, content[i], (box[0, 1], box[1, 0]), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                             fontScale=1, color=(0, 0, 255))
        canvas[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1]] = np.minimum(im,
                                                                      canvas[box[0, 0]:box[1, 0], box[0, 1]:box[1, 1],
                                                                      :])

    return canvas


def load_test_input(ims, cur_label, config):
    down_symbols = config.DOWN_SMALL_SYMBOLS
    up_symbols = config.UP_SMALL_SYMBOLS

    pos_embed_ids = []
    for i, symbol in enumerate(cur_label):
        if symbol in down_symbols:
            pos_embed_id = 1
            h, w = ims[i].shape
            if h > 25:
                ims[i] = cv2.resize(ims[i], (0, 0), fx=0.5, fy=0.5)
        elif symbol in up_symbols:
            pos_embed_id = 2
        else:
            pos_embed_id = 0
        pos_embed_ids.append(pos_embed_id)

    sample_len = len(cur_label)
    test_ims = []
    out_ims = []
    shapes = []
    max_h = 0
    max_w = 0

    for i in range(sample_len):
        raw_im = ims[i]
        h, w = raw_im.shape
        test_ims.append(np.copy(raw_im))

        im = 255 - otsu_thresh(cv2.resize(pad2square(raw_im), (config.in_im_height, config.in_im_width)))
        shapes.append([h, w])
        out_ims.append(im / 255.)
        max_h = max(h, max_h)
        max_w = max(w, max_w)

    slope = np.random.normal(loc=config.slope_mean, scale=config.slope_std)
    base_gap = int(np.random.normal(loc=config.base_gap_mean, scale=config.base_gap_std))
    if base_gap < 0:
        base_gap = 0

    shapes = np.asarray(shapes).astype(np.float32)
    shapes[:, 0] = (shapes[:, 0] - config.h_mean) / config.h_std
    shapes[:, 1] = (shapes[:, 1] - config.w_mean) / config.w_std

    seq_len = len(out_ims)
    out_ims = torch.from_numpy(np.asarray(out_ims)).unsqueeze(dim=0).float()
    seq_len = torch.from_numpy(np.asarray([seq_len]))
    shapes = torch.from_numpy(shapes).unsqueeze(dim=0)
    pos_embed_ids = torch.from_numpy(np.array(pos_embed_ids)).unsqueeze(dim=0)
    return out_ims, test_ims, seq_len, shapes, pos_embed_ids, slope, base_gap, max_h, max_w


def load_model():
    config = ProjectConfig()
    model = LayoutModel(config)
    model, iter_num, best_acc = load_last_checkpoint(model,
                                                     config.CHECK_POINT_FOLDER,
                                                     config.MODEL_NAME,
                                                     config.USE_GPU)
    model = model.eval()
    return model


def inference_im_model(ims, cur_label, start_lu_pos, model):
    '''
    train model
    :return: None
    '''

    config = ProjectConfig()

    ims, test_ims, seq_len, shapes, pos_embed_ids, slope, base_gap, max_h, max_w = load_test_input(ims, cur_label,
                                                                                                   config)
    # base_gap = int((base_gap - config.base_gap_mean) / config.base_gap_std)
    # base_gap = min(base_gap, 15)
    normalized_gap = torch.from_numpy(np.asarray([(base_gap - config.base_gap_mean) / config.base_gap_std])).float()
    normalized_slope = torch.from_numpy(np.asarray([(slope - config.slope_mean) / config.slope_std])).float()
    if config.USE_GPU:
        ims = ims.cuda()
        seq_len = seq_len.cuda()
        shapes = shapes.cuda()
        normalized_gap = normalized_gap.cuda()
        normalized_slope = normalized_slope.cuda()
        pos_embed_ids = pos_embed_ids.cuda()
    preds = model.forward(ims, seq_len, shapes, pos_embed_ids, normalized_gap, normalized_slope, config)
    preds = preds[0].detach().cpu().numpy()
    normalized_h_centers_diff = preds[:, 0]
    normalized_gap_diffs = preds[:, 1]

    h_centers_diff = normalized_h_centers_diff * config.center_shift_std + config.center_shift_mean
    gap_diffs = normalized_gap_diffs * config.gap_shift_std + config.gap_shift_mean

    lu_poses, next_pos = form_im(start_lu_pos, h_centers_diff, gap_diffs, slope, base_gap, test_ims, max_h, max_w)
    return lu_poses, next_pos
