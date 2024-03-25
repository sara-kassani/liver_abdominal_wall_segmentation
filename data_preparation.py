# -*- coding: utf-8 -*-
from utils import rename_files, convert_dcm2png, convert_nii2png
import os
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    in_dir="files/images/"
    out_dir="data_processed/images/"
    convert_dcm2png(in_dir, out_dir)   
    
    
    in_dir="files/masks/"
    out_dir= "data_processed/masks/"
    convert_nii2png(in_dir, out_dir, is_label=True)   
    
    ## convert 3 class wall layers into one class
    in_dir= "data_processed/masks_multi/"
    out_dir= "data_processed/masks_binary/"
    all_files= os.listdir(in_dir)
    wall_files= os.listdir(in_dir)
    
    ### create multiclass to binary mask
    for fname in wall_files:
        fpath = os.path.join(in_dir, fname)
        mask = cv2.imread(fpath, -1)
        print(np.unique(mask))
        mask[mask == 3] = 2  # 1 is for liver and 2 for abdomial wall layer
        mask[mask == 4] = 2
        save_path = os.path.join(out_dir, fname)
        cv2.imwrite(save_path, mask) 
        
    ## sanity check
    for fname in os.listdir(out_dir):
        binary_mask = cv2.imread(os.path.join(out_dir, fname), -1)
        plt.imshow(binary_mask, cmap= "gray")
        plt.show()
        
    ## fill small holes inthe abd wall areas
    filled_dir= "data_processed/masks_binary_filled/"
    binary_files= os.listdir(out_dir)
    
    for fname in binary_files:
        img_path= os.path.join(out_dir, fname)
        binary_mask= cv2.imread(img_path, -1) 
        filled_mask= ndimage.binary_fill_holes(binary_mask)
        filled_mask= np.array(filled_mask, dtype= np.uint8)
    #     print(np.unique(filled_mask))
        save_path= os.path.join(filled_dir, fname)
        cv2.imwrite(save_path, filled_mask)
        
    ## sanity check
    for fname in os.listdir(filled_dir):
        binary_mask = cv2.imread(os.path.join(filled_dir, fname), -1)
        plt.imshow(binary_mask, cmap= "gray")
        plt.show()