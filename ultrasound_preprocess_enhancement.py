# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import copy
import glob
from skimage.filters import unsharp_mask
from scipy import ndimage
from utils import combine_masks


def remove_black_bk_img_mask(img_indir, img_outdir):
    fnames_img= os.listdir(img_indir)
    for fname in fnames_img:
        img_gray= cv2.imread(os.path.join(img_indir, fname), 0)

        img= copy.copy(img_gray)
        ret, thresh= cv2.threshold(img_gray, 0, 255, 0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contoured_img= cv2.drawContours(cv2.imread(os.path.join(img_indir, fname)), contours, -1, (0, 255, 0), 3)

        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)

        mask = np.zeros(img.shape[:2],np.uint8)
        cv2.drawContours(mask, [cnt],-1, 255, -1)
        dst = cv2.bitwise_and(img, img, mask=mask)
        
#         plt.imshow(dst, cmap= 'gray')
#         plt.title(dst.shape)
#         plt.show()

        img_savepath= os.path.join(img_outdir, fname)
        cv2.imwrite(img_savepath, dst)
        
        
def unsharp_mask(in_dir, out_dir, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory '{in_dir}' not found.")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    file_paths = glob.glob(in_dir + "/*.png")
    
    if not file_paths:
        raise FileNotFoundError(f"No PNG files found in '{in_dir}'.")
    
    for file_path in file_paths:
        input_img = cv2.imread(file_path)
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, kernel_size, sigma)
        sharpened = float(amount + 1) * gray_img - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(gray_img - blurred) < threshold
            np.copyto(sharpened, gray_img, where=low_contrast_mask)
        save_path = os.path.join(out_dir, os.path.basename(file_path)[:-4] + '.png')
        cv2.imwrite(save_path, sharpened)



def clahe2d(in_dir, out_dir):
    file_paths= glob.glob(in_dir + "/*.png")
    
    for file_path in file_paths:
        input_img= cv2.imread(file_path, 0)
#         gray_img= cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        gray_img= input_img.astype(np.uint8)
        clahe= cv2.createCLAHE(clipLimit= 2, tileGridSize= (20, 20))
        clahe_img= clahe.apply(gray_img)
#         print(clahe_img.shape)
        save_path= os.path.join(out_dir, os.path.basename(file_path)[:-4] + '.png')
        cv2.imwrite(save_path, clahe_img)
        
        
def laplac_filter(in_dir, out_dir):
    file_paths= glob.glob(in_dir + "/*.png")

    for file_path in file_paths:
        input_img= cv2.imread(file_path, 0)
        laplac_img = ndimage.gaussian_laplace(input_img, sigma=0.45, mode='nearest')

        save_path= os.path.join(out_dir, os.path.basename(file_path)[:-4] + '.png')
        cv2.imwrite(save_path, laplac_img)
        
def minpool_norm(in_dir, out_dir, sigmaX= 10):
    file_paths= glob.glob(in_dir + "/*.png")

    for file_path in file_paths:
        in_img= cv2.imread(file_path, -1)
        in_img= cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
        minpool_img= cv2.addWeighted(in_img, 4, cv2.GaussianBlur(in_img, (0, 0), sigmaX), -4, 128)
   
        save_path= os.path.join(out_dir, os.path.basename(file_path)[:-4] + '.png')
        cv2.imwrite(save_path, minpool_img)               

  

if __name__ == "__main__":

    ## process abdominal wall images
    img_indir="abdwall_binary/images/"
    img_outdir="abdwall_binary/images_cleaned/"
    # ----------------------------------
    remove_black_bk_img_mask(img_indir=img_indir, img_outdir=img_outdir)
    # ----------------------------------
    unsharp_mask(in_dir="abdwall_binary/images_cleaned/", out_dir="abdwall_binary/images_sharpen/")
    clahe2d(in_dir="abdwall_binary/images_cleaned/", out_dir="abdwall_binary/images_clahe/")
    clahe2d(in_dir="abdwall_binary/images_sharpen/", out_dir="abdwall_binary/images_sharpen_clahe/")
    clahe2d(in_dir="abdwall_binary/images_clahe/", out_dir="abdwall_binary/images_2clahe/")
    # minpool_norm(in_dir="abdwall_binary/images_cleaned/", out_dir="abdwall_binary/images_mipool/")
    
    # ----------------------------------
    # process liver images    
    mask_paths= glob.glob("data_processed/masks/*.png")
    for mask_path in mask_paths:
        mask= cv2.imread(mask_path, -1)
        mask[mask == 255] = 1
        mask_path= os.path.join("data_processed/masks_onehot/", os.path.basename(mask_path))
        cv2.imwrite(mask_path, mask)
        
    # ----------------------------------
    ## combine liver (1) and abdominal wall (2) masks
    combine_masks(liver_dir="data/liver_mask_png_125/", 
              abd_dir="data/abdwall_masks_png_125/", 
              dest_dir="data/combined_masks/")
    
    