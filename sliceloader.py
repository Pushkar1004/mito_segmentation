import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
from utils import create_patches, create_patches_list
from read_write_mrc import read_mrc, write_mrc
import numpy as np


class SliceLoader(Dataset):
    def __init__(self, data_root_dir=None, images_dir= "images", labels_dir="labels"):
        
        assert data_root_dir is not None, "Please pass the correct the data_root_dir"
        
        images_list = []
        
        for path in os.listdir(os.path.join(data_root_dir, images_dir)):
            images_list.extend(glob.glob(os.path.join(data_root_dir, images_dir, path)))
        images_list.sort()
            
        labels_list = []
        
        for path in os.listdir(os.path.join(data_root_dir, labels_dir)):
            labels_list.extend(glob.glob(os.path.join(data_root_dir, labels_dir, path)))
        labels_list.sort()
            
            
        assert (len(labels_list) == len(images_list)), "lengths of labels and images should be same."
            
        self.images_list = images_list
        self.labels_list = labels_list
        

        
        
    def __getitem__(self, item):
        image = read_mrc(os.path.join(self.images_list[item])) 
        image = np.stack((image,)*1, axis = 0)  # or we can use image.unsqueeze
        
        label = read_mrc(os.path.join(self.labels_list[item]))
        label = np.where((label == 5.0), torch.ones(label.shape), torch.zeros(label.shape))
        label = np.stack((label,np.where((label == 0), np.ones(label.shape), np.full(label.shape, 0) )), 0)
        
        # label = np.stack((label,)*1, axis = 0)  # or we can use image.unsqueeze
         
        return image, label
        

    def __len__(self):
        return len(self.images_list)