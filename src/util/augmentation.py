from skimage import io
from skimage import transform as tf
import numpy as np
import cv2
import random

from src.util.image_util import otsu_thresh
import pdb


def random_shear_rotation(img, shear_factor=0.1, ang_range=10):
    """Randomly shears and rotates an image in both the directions 
    
    Parameters
    ----------
    shear_factor: float 
    ang_range: int    
    Returns
    -------    
    numpy.ndaaray
        Sheared and rotated image in the numpy format of shape `HxW`           
    """
    # Adding rotation to the image
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    w, h = img.shape[1], img.shape[0]
    Rot_M = cv2.getRotationMatrix2D((w / 2, h / 2), ang_rot, 1)
    img = cv2.warpAffine(img, Rot_M, (w, h), borderValue=255)

    # Adding shear to the image
    shear_factor = random.uniform(-shear_factor, shear_factor)
    w, h = img.shape[1], img.shape[0]
    if shear_factor < 0:
        img = cv2.flip(img, 1)
    M = np.array([[1, abs(shear_factor), 0], [abs(shear_factor), 1, 0]])
    nW = img.shape[1] + abs(shear_factor * img.shape[0])
    nH = img.shape[0] + abs(shear_factor * img.shape[1])
    img = cv2.warpAffine(img, M, (int(nW), int(nH)), borderValue=255)
    if shear_factor < 0:
        img = cv2.flip(img, 1)
    img = cv2.resize(img, (w, h))
    return img


def morphological_operations(img):
    """Randomly adding morphological operations on the image
    
    Parameters
    ----------
    img: ndarray
    Returns
    -------    
    numpy.ndaaray
       Morphed image in the numpy format 
    """
    operations = ['dilate', 'erode']
    selected_operation = random.choice(operations)
    selected_operation = 'erode'
    if selected_operation == 'dilate':
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    elif selected_operation == 'erode':
        kernel_szie = random.choice([1, 3])
        kernel = np.ones((kernel_szie, kernel_szie), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    # img = otsu_thresh(img)
    return img
