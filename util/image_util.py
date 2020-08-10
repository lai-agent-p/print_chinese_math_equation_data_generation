import cv2
import numpy as np
import random
from math import ceil, cos, sin, pi
from skimage.filters import threshold_sauvola

def safe_savola(im):
    otsu_im = 255 - otsu_thresh(im)
    savola_im = 255 - savola(im)
    if np.sum(otsu_im) > 1.3 * np.sum(savola_im):
        return 255 - otsu_im
    else:
        return 255 - savola_im

def savola(img):
    thresh_sauvola = threshold_sauvola(img, window_size=25)
    binary_sauvola = img > thresh_sauvola
    ret_sauvola = binary_sauvola.astype(np.uint8) * 255
    return ret_sauvola

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


def otsu_thresh(im):
    '''
    binarize image with otsu algorithm
    :param im: uint8 numpy array
    :return: binarized image as uint8 numpy array
    '''
    to_binarize = np.copy(np.uint8(im))
    ret2, th2 = cv2.threshold(to_binarize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2


def rotate_and_pad(im, theta):
    '''

    :param im:
    :param theta:
    :return:
    '''
    h, w = im.shape
    theta_pi = theta / 180 * pi
    ud_pad = abs(int(ceil(h / 2 * cos(theta_pi) + w / 2 * sin(theta_pi) - h / 2)))
    lr_pad = abs(int(ceil(w / 2 * cos(theta_pi) + h / 2 * sin(theta_pi) - w / 2)))
    im = np.pad(im, ((ud_pad, ud_pad), (lr_pad, lr_pad)), 'constant', constant_values=255)
    M = cv2.getRotationMatrix2D((w / 2 + lr_pad, h / 2 + ud_pad), theta, 1)
    dst = cv2.warpAffine(im, M, (w + 2 * lr_pad, h + 2 * ud_pad), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    dst = cv2.resize(dst, (w, h))
    return dst


def probablity_binarization(im, var_param=1, mode='darker', intensity=0.5):
    #     exag = 0.1
    im = (im / 255)

    if mode == 'darker':
        prob_pixels = np.logical_and(im < 0.99, im > intensity)
        im[im >= 0.99] = 1.
        im[im <= intensity] = 0.
    elif mode == 'lighter':
        prob_pixels = np.logical_and(im < 1 - intensity, im > 0.01)
        im[im >= 1 - intensity] = 1.
        im[im <= 0.01] = 0.
    random_map = np.random.rand(im.shape[0], im.shape[1])
    im[prob_pixels] = random_map[prob_pixels] < im[prob_pixels]
    im = np.array(im * 255, dtype=np.uint8)
    return im


def cut_and_move(im, n_cuts=30, n_pixel=1):
    cuts = sorted(np.random.rand(n_cuts))
    h, w = im.shape
    for i in range(len(cuts) - 1):
        if i % 2:
            im[0:h - 1, int(w * cuts[i]):int(w * cuts[i + 1])] = im[1:h, int(w * cuts[i]):int(w * cuts[i + 1])]
    #     print(cuts)
    return im

def rotate_image(image, angle):
    #     image_center = tuple(np.array(image.shape[1::-1]) / 2)
    h, w = image.shape
    random_place = (random.random() * w, random.random() * h)
    rot_mat = cv2.getRotationMatrix2D(random_place, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    return result

def gen_messy_line(length, thickness):
    length*=4
    normlized_length = length / thickness
    rotate_angle_dist = [0.0, 0.03, 0.05, 0.1, 0.2, 0.3]
    fake_thickness_dist = [0.4, 0.6, 0.8, 1.1, 1.2, 1.4]
    cuts_dist = [0, 5, 10, 20]
    fake_width = random.choice(fake_thickness_dist)
    line = np.ones((max(10, (int(normlized_length / 20))), int(normlized_length))) * 255
    if fake_width <= 1:
        line[max(5, int(normlized_length / 40)):max(5, int(normlized_length / 40)) + 1, int(normlized_length * 0):int(normlized_length)] = 0
    else:
        line[max(5, int(normlized_length / 40)):max(5, int(normlized_length / 40)) + 2, int(normlized_length * 0):int(normlized_length)] = 0
    line = 255 - rotate_image(line, random.choice(rotate_angle_dist))
    if fake_width <= 1:
        line = probablity_binarization(line, mode='lighter', intensity=fake_width)
    else:
        line = probablity_binarization(line, mode='lighter', intensity=2 - fake_width)
    #     print(fake_width)

    line = cut_and_move(line, n_cuts=random.choice(cuts_dist))
    h, w = line.shape
    line = cv2.resize(line, (length//4, int(h / w * length)//4), interpolation=cv2.INTER_CUBIC)
    col_sums = np.sum(line, axis=1)
    row_start = np.where(col_sums)[0][0]
    rev_col_sum = np.flip(col_sums, axis=0)
    rev_end = np.where(rev_col_sum)[0][0]
    row_end = col_sums.shape[0] - rev_end
    line = line[row_start: row_end, :]
    line = otsu_thresh(line)
    # cv2.imwrite('line.png', line)
    #     show_img_np(line)
    return line

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