# -*- coding: utf-8 -*-
import os
import pydicom
import glob
import numpy as np
import cv2
from PIL import Image
import nibabel as nib

def rename_files(in_dir):
    for root, dirs, fnames in os.walk(in_dir):
        for fname in fnames:
            if fname.endswith(".nii.gz"):
                filename= fname[:-7]
                nii_path= os.path.join(root, fname)
                for file in os.listdir(os.path.dirname(nii_path)):
                    if file.endswith(".dcm"):
                        new_name= filename+".dcm"
                        new_path= os.path.join(os.path.dirname(nii_path), new_name)
                        old_path= os.path.join(os.path.dirname(nii_path), file)
                        print(fname, " | ",  file)
                        print(old_path, "\n", new_path, "\n =============================")
                        os.rename(old_path, new_path)

def convert_dcm2png(in_dir, out_dir):
    fnames= os.listdir(in_dir)
    for fname in fnames:
        if fname.endswith(".dcm"):
            f_name= os.path.basename(fname)
            filename= f_name[:-4]+".png"
            print(fname)
            dcm_data = pydicom.dcmread(os.path.join(in_dir, fname))
            dcm_data = dcm_data.pixel_array.astype(float)
            rescaled_image = (np.maximum(dcm_data,0)/dcm_data.max())*255 # float pixels
            final_image = np.uint8(rescaled_image) # integers pixels
            final_image = Image.fromarray(final_image)
            save_path= os.path.join(out_dir, filename)
            print(f"Processing: {f_name}, saved at: {save_path}")
            final_image.save(save_path)
        
def convert_tiff2png(tiff_path, png_path):
    img = Image.open(tiff_path)
    img.save(png_path, 'PNG')        
        
def convert_nii2png(in_dir, out_dir, is_label=False):
    file_paths= glob.glob(in_dir + "/*.nii")
    
    for file_path in file_paths:
        nii_img = nib.load(file_path).get_fdata()
        num = 0
        for i in range(nii_img.shape[2]):
            nii_arr = nii_img[:, :, i].astype(np.uint8)
            
            if is_label:
                nii_arr = np.where(nii_arr == 1, 255, nii_arr)
            

#             save_path= os.path.join(out_dir, os.path.basename(file_path)[:-4] + '_slice' + str(num) + '.png')
            final_image = np.uint8(nii_arr) # integer pixels
            final_image = np.moveaxis(final_image, -1, 0)
            final_image = Image.fromarray(final_image)
#             print(np.shape(final_image))
            
            save_path= os.path.join(out_dir, os.path.basename(file_path)[:-4] + '.png')
            print(f"Processing: {os.path.basename(file_path)}, saved at: {save_path}")
            final_image.save(save_path)

            num += 1

def combine_masks(liver_dir, abd_dir, dest_dir):
    liver_masks = os.listdir(liver_dir)
    abd_masks = os.listdir(abd_dir)
    
    # Ensure we have the same number of masks in each folder
    assert len(liver_masks) == len(abd_masks), "Number of masks in both folders must be the same"

    for liver_mask_file, abd_mask_file in zip(liver_masks, abd_masks):
        liver_path = os.path.join(liver_dir, liver_mask_file)
        abd_path = os.path.join(abd_dir, abd_mask_file)


        liver_mask = cv2.imread(liver_path, cv2.IMREAD_GRAYSCALE)
        abd_mask = cv2.imread(abd_path, cv2.IMREAD_GRAYSCALE)

        # Ensure the masks are binary (0 and 255)
        liver_mask = np.where(liver_mask>0, 255, 0)
        abd_mask = np.where(abd_mask>0, 127, 0)

        combined_mask = cv2.bitwise_or(liver_mask.astype(np.uint8), abd_mask.astype(np.uint8))

        combined_path = os.path.join(dest_dir, liver_mask_file)
        cv2.imwrite(combined_path, combined_mask)
            
            
            
      