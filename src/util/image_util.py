import cv2
import numpy as np
from lailib.image.resize import resize_height_keep_ratio
from math import sin, cos, ceil, pi
from skimage import morphology
from PIL import ImageFont, ImageDraw, Image
import os
import math
import random
from skimage.filters import threshold_sauvola
from scipy.ndimage import interpolation
import pdb


def savola(img):
    thresh_sauvola = threshold_sauvola(img, window_size=25)
    binary_sauvola = img > thresh_sauvola
    ret_sauvola = binary_sauvola.astype(np.uint8) * 255
    return ret_sauvola


def otsu_thresh(im):
    '''
    binarize image with otsu algorithm
    :param im: uint8 numpy array
    :return: binarized image as uint8 numpy array
    '''
    to_binarize = np.copy(np.uint8(im))
    ret2, th2 = cv2.threshold(to_binarize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def hor_crop(im, binarized):
    '''

    crop image only on horizontal direction

    Args:
        im: uint8ndarray
            input image to be cropped
        binarized: uint8ndarray
            binarized version of im,
            at least this binarized image should have
             the same shape as im
    Returns:
        cropped image
    '''

    row_sums = np.sum(binarized, axis=0)
    col_start = np.where(row_sums)[0][0]
    rev_row_sum = np.flip(row_sums, axis=0)
    rev_end = np.where(rev_row_sum)[0][0]
    col_end = row_sums.shape[0] - rev_end

    cropped = im[:, col_start:col_end]

    return cropped


def resize_on_long_aixs(full_im, cropped_im):
    full_h, full_w = full_im.shape
    cropped_h, cropped_w = cropped_im.shape
    full_ratio = full_h / full_w
    cropped_ratio = cropped_h / cropped_w
    if full_ratio > cropped_ratio:
        # based on w
        cropped_im = cv2.resize(cropped_im, (full_w, int(full_w / cropped_w * cropped_h)))
    else:
        # based on h
        cropped_im = cv2.resize(cropped_im, (int(full_h / cropped_h * cropped_w), full_h))
    return cropped_im


def resize_image(image_cv, new_height, **kwargs):
    '''
    resize image with given height, keep ratio
    :param image_cv: image to be resized
    :param new_height: new height of the output image
    :param inte_method: interpolation method
    :return: resized image
    '''
    (height, width) = image_cv.shape
    new_width = int(float(width) * new_height / float(height))
    image_cv = cv2.resize(image_cv, (new_width, new_height), **kwargs)
    return image_cv


def crop_and_padding(image_cv, padding=0, new_height=None, binarized=False):
    '''
    crop and padding an image, may resize image to a height
    :param image_cv: input image, uint8 numpy array, assumed to be gray scale
                     (if binarized set to true, image will be kept binarized)
    :param padding: padding to each side
    :param new_height: new_height of image if required
    :return: cropped and potentially padded and resized image (uint8)
    '''
    col_sums = np.sum(image_cv, axis=1)
    row_start = np.where(col_sums)[0][0]
    rev_col_sum = np.flip(col_sums, axis=0)
    rev_end = np.where(rev_col_sum)[0][0]
    row_end = col_sums.shape[0] - rev_end

    row_sums = np.sum(image_cv, axis=0)
    col_start = np.where(row_sums)[0][0]
    rev_row_sum = np.flip(row_sums, axis=0)
    rev_end = np.where(rev_row_sum)[0][0]
    col_end = row_sums.shape[0] - rev_end

    cropped = image_cv[row_start:row_end, col_start:col_end]

    # TODO add rand noise to height
    if new_height:
        cropped = resize_image(cropped, new_height)
    if binarized:
        ret2, cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height, width = cropped.shape
    final_out = cropped
    if padding:
        final_out = np.zeros((height + 2 * padding, width + 2 * padding))
        final_out[padding:-padding, padding:-padding] = cropped
    return final_out


def process_thin_img(im):
    kernel = np.ones((3, 3), np.uint8)
    im = np.pad(im, ((3, 3), (3, 3)), 'maximum')
    im = 255 - ((255 - im) * 0.8).astype(np.uint8)
    im = cv2.erode(im, kernel, iterations=10)
    im = blur_image(im)
    return im


def blur_image(im):
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(im, -1, kernel)
    return dst


def flip_tuple(coord):
    return (coord[1], coord[0])


def erode_plus(orig_img, target_thickness=2):
    orig_img = otsu_thresh(orig_img)
    orig_img = 255 - orig_img
    orig_img_copy = orig_img.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = morphology.skeletonize(orig_img > 0)
    erode_baseline = None
    thickness = 0

    while True:
        # does not support 0 or 1 thickness yet
        thickness += 1
        orig_sum = np.sum(orig_img > 0)
        # erode but keep skel
        orig_img = cv2.dilate(orig_img, element, iterations=1)
        orig_img = cv2.erode(orig_img, element, iterations=2)
        orig_img = cv2.bitwise_or(np.array(skel, dtype=np.uint8) * 255, orig_img)
        # calcualte erode rate, if too little, it means it's almost skel already
        erode_rate = orig_sum - np.sum(orig_img > 0)
        if erode_baseline is None:
            erode_baseline = erode_rate
        else:
            if (erode_rate < erode_baseline / 2) or (erode_rate < 10):
                break
    # print(thickness)
    orig_img = orig_img_copy.copy()
    if thickness - target_thickness + 1 < 0:
        return orig_img, thickness
    #         raise NotImplementedError("wait")
    # print(thickness - target_thickness + 1)
    for i in range(thickness - target_thickness + 1):
        orig_img = cv2.dilate(orig_img, element, iterations=1)
        orig_img = cv2.erode(orig_img, element, iterations=2)
        orig_img = cv2.bitwise_or(np.array(skel, dtype=np.uint8) * 255, orig_img)
    # orig_img = blur_image(255 - orig_img)
    orig_img = 255 - orig_img
    return orig_img, thickness


def fix_lr_boundary(image_cv, padding=15):
    '''
    fix left and right padding for an image
    :param image_cv: input image, uint8 numpy array, assumed to be gray scale
                     (if binarized set to true, image will be kept binarized)
    :return: cropped and potentially padded and resized image (uint8)
    '''
    up_padding = 10
    down_padding = 15
    _, binarized = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_h, im_w = image_cv.shape
    col_sums = np.sum(binarized, axis=1)
    row_start = np.where(col_sums)[0][0]
    rev_col_sum = np.flip(col_sums, axis=0)
    rev_end = np.where(rev_col_sum)[0][0]
    row_end = col_sums.shape[0] - rev_end
    row_sums = np.sum(binarized, axis=0)
    col_start = np.where(row_sums)[0][0]
    rev_row_sum = np.flip(row_sums, axis=0)
    rev_end = np.where(rev_row_sum)[0][0]
    col_end = row_sums.shape[0] - rev_end
    if (row_end - row_start) > 0.3 * im_h:
        cropped = image_cv[row_start:row_end, col_start:col_end]
    else:
        cropped = image_cv[:, col_start:col_end]
    height, width = cropped.shape
    rel_up_padding = ceil(up_padding * height / 75)
    rel_down_padding = ceil(down_padding * height / 75)
    rel_lr_padding = ceil(padding * height / 75)
    final_out = cropped
    if padding:
        if (row_end - row_start) > 0.3 * im_h:
            final_out = np.zeros((height + rel_up_padding + rel_down_padding, width + 2 * rel_lr_padding))
            final_out[rel_up_padding:-rel_down_padding, rel_lr_padding:-rel_lr_padding] = cropped
        else:
            final_out = np.zeros((height, width + 2 * rel_lr_padding))
            final_out[:, rel_lr_padding:-rel_lr_padding] = cropped
    return final_out


def estimate_skew_angle(image, angles):
    """estimate skew angle """
    estimates = []
    for a in angles:
        v = np.mean(interpolation.rotate(image, a, order=0, mode='constant'), axis=1)
        v = np.var(v)
        estimates.append((v, a))
    _, a = max(estimates)
    return a


def estimate_skew(flat, bignore, maxskew, skewsteps):
    ''' estimate skew angle and rotate'''
    d0, d1 = flat.shape
    o0, o1 = int(bignore * d0), int(bignore * d1)
    flat = 1 - flat
    est = flat[o0:d0 - o0, o1:d1 - o1]
    ma = maxskew
    ms = int(2 * maxskew * skewsteps)
    angle = estimate_skew_angle(est, np.linspace(-ma, ma, ms + 1))
    flat = interpolation.rotate(flat, angle, mode='constant', reshape=0)
    flat[flat < 0] = 0.
    flat = 1 - flat
    return flat, angle


def deskew(image):
    bignore = 0.1
    maxskew = 2
    skewsteps = 8
    image = image.copy() / 255.0
    image, angle = estimate_skew(image, bignore, maxskew, skewsteps)
    return (image * 255).astype(np.uint8)


def pad2square(im):
    h, w = im.shape
    if h > w:
        pad_w = h - w
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad
        im = np.pad(im, ((0, 0), (left_pad, right_pad)), constant_values=((255, 255), (255, 255)))
    else:
        pad_h = w - h
        up_pad = pad_h // 2
        down_pad = pad_h - up_pad
        im = np.pad(im, ((up_pad, down_pad), (0, 0)), constant_values=((255, 255), (255, 255)))
    return im
