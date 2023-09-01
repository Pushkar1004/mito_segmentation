import numpy as np
import skimage
from read_write_mrc import read_mrc, write_mrc



def normalisation(img):
    norm_img = (img-img.min())/(img.max()-img.min())
    return norm_img


def adapt_hist_eq(img):
    eq_img = np.where(img != 0, skimage.exposure.equalize_adapthist(img), np.zeros(shape = img.shape) )
    return eq_img

def hist_eq(img):
    eq_img = skimage.exposure.equalize_hist(img)
    return eq_img


def padding(img):
    pad_img = np.pad(img, ( (704-img.shape[0],0) , (704-img.shape[1],0) , (704-img.shape[2],0) ))
    return pad_img



def mask_crop(image, labels):
    mask = np.where(labels != 0, np.full(labels.shape, 255), np.zeros(shape = labels.shape))
    masked = np.where(mask == 255, image, np.zeros(shape = image.shape))
    return (masked,labels)
