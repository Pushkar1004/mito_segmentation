import numpy as np
import skimage
from read_write_mrc import read_mrc, write_mrc
from patchify import patchify, unpatchify

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



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def create_patches(img):
    patches_x = patchify(img, (64,704,704), step= (32,32,32))
    patches_x = patches_x.squeeze((1,2))
    tmp = np.rot90(img, axes=(0,1))
    patches_y = np.flip(tmp, axis = 0)
    patches_y = patchify(patches_y, (64,704,704), step = 32)
    patches_y = patches_y.squeeze((1,2))                 
    tmp = np.rot90(tmp, axes=(2,0))
    patches_z = patchify(tmp, (64 ,704 ,704), step = 32)
    patches_z = patches_z.squeeze((1,2))    
    return np.concatenate((patches_x, patches_y, patches_z), axis = 0)



def create_patches_list(img):
    
    patches = []
    
    patches_x = patchify(img, (64,704,704), step= (32,32,32))
    patches_x = patches_x.squeeze((1,2))
    
    for i in range(patches_x.shape[0]):
        patches.append(patches_x[i])
    
    tmp = np.rot90(img, axes=(0,1))
    patches_y = np.flip(tmp, axis = 0)
    patches_y = patchify(patches_y, (64,704,704), step = 32)
    patches_y = patches_y.squeeze((1,2))
    
    
    for i in range(patches_y.shape[0]):
        patches.append(patches_y[i])
    
    tmp = np.rot90(tmp, axes=(2,0))
    patches_z = patchify(tmp, (64 ,704 ,704), step = 32)
    patches_z = patches_z.squeeze((1,2))  
    
    
    for i in range(patches_z.shape[0]):
        patches.append(patches_z[i])
    
    return np.array(patches)