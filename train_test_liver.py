# -*- coding: utf-8 -*-
# !python -c "import monai" || pip install -q "monai-weekly[gdown, nibabel, tqdm, ignite]"
# !python -c "import matplotlib" || pip install -q matplotlib
# !pip install "monai[einops]"
import os
import sys
import cv2
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import array_to_img
import torch
import monai
from monai.utils import first, set_determinism
from monai.transforms import Compose, AddChanneld, ScaleIntensityd, Resized, SpatialPad, AsDiscrete, ToTensor, LoadImaged, Activations, Rotate90d,ResizeWithPadOrCropd, EnsureChannelFirstd, SpatialPadd, EnsureTyped, ConcatItemsd, AddChanneld, AsDiscreted, ScaleIntensity, Flipd
from monai.networks.nets import UNet, VNet, AttentionUnet, BasicUNetPlusPlus, UNETR, SwinUNETR
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss, TverskyLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.utils import set_determinism
from monai.data import DataLoader, decollate_batch, PILReader
from monai.handlers.utils import from_engine
from torchvision.utils import save_image
from monai.config import print_config

import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
print('Python:', sys.version)
print('Numpy:', np.__version__)
print('torch', torch.__version__)
#------------------------------------------------------------------------------
print_config()
#------------------------------------------------------------------------------
set_determinism(seed=777)
#------------------------------------------------------------------------------
data_dir= "r01_liveronly_patientlevel/"
train_dir= os.path.join(data_dir, "train/")
validation_dir= os.path.join(data_dir, "validation/")
test_dir= os.path.join(data_dir, "test/")

train_images = sorted(glob.glob(train_dir + "images_clahe/*.png"))
train_masks = sorted(glob.glob(train_dir + "masks/*.png"))

valid_images = sorted(glob.glob(validation_dir + "images_clahe/*.png"))
valid_masks = sorted(glob.glob(validation_dir + "masks/*.png"))

test_images = sorted(glob.glob(test_dir + "images_clahe/*.png"))
test_masks = sorted(glob.glob(test_dir + "masks/*.png"))

# '''--------------------split data into train and validation and test sets--------------------'''

# train_temp_images, test_images, train_temp_masks, test_masks = train_test_split(image_paths, mask_paths, test_size=0.2)
# train_images, valid_images, train_masks, valid_masks = train_test_split(train_temp_images, train_temp_masks, test_size=0.2)

# print(f'#training samples: {np.shape(train_images)[0]}, #validation samples: {np.shape(valid_images)[0]}, #test samples: {np.shape(test_images)[0]}')


#------------------------------------------------------------------------------
img_check= cv2.imread(train_images[0], -1)
print(img_check.shape)
#------------------------------------------------------------------------------

train_files = [{"img": image_name, "label": label_name}
               for image_name, label_name in zip(train_images, train_masks)]

val_files = [{"img": image_name, "label": label_name}
               for image_name, label_name in zip(valid_images, valid_masks)]

test_files = [{"img": image_name, "label": label_name}
               for image_name, label_name in zip(test_images, test_masks)]

#------------------------------------------------------------------------------
size_factor= 28
img_size = ((32*size_factor), (32*size_factor))
# img_size=(873, 1164)
from monai.transforms import LoadImaged, EnsureChannelFirstd, ConcatItemsd

train_transforms = Compose([
    LoadImaged(keys=["img", "label"], reader=PILReader),
    EnsureChannelFirstd(keys=["img", "label"]),
    ScaleIntensityd(keys=["img"], allow_missing_keys=True),
    Resized(keys=["img", "label"], spatial_size=img_size),
#     ResizeWithPadOrCropd(keys=["img", "label"], spatial_size=img_size),
    EnsureTyped(keys=["img", "label"]),
    Flipd(keys=["img", "label"], spatial_axis=0),
    Rotate90d(keys=["img", "label"], k=3),
    EnsureTyped(keys=["img", "label"]),  
#     Resized(keys=["img", "label"], spatial_size=img_size, mode="bilinear"), 
#     SpatialPadd(keys=["img", "label"], spatial_size=1024),,    
])

val_transforms = Compose([
    LoadImaged(keys=["img", "label"], reader=PILReader),
    EnsureChannelFirstd(keys=["img", "label"]),
    ScaleIntensityd(keys=["img"], allow_missing_keys=True),
    Resized(keys=["img", "label"], spatial_size=img_size),
#     ResizeWithPadOrCropd(keys=["img", "label"], spatial_size=img_size),
    EnsureTyped(keys=["img", "label"]),
    Flipd(keys=["img", "label"], spatial_axis=0),
    Rotate90d(keys=["img", "label"], k=3),
    EnsureTyped(keys=["img", "label"]),  
#     Resized(keys=["img", "label"], spatial_size=img_size, mode="bilinear"), 
#     SpatialPadd(keys=["img", "label"], spatial_size=1024),    
])               

# If you want to preserve as much of the original image information as possible and are willing to accept some distortion, 
# then Resized is a good option. If you want to preserve the aspect ratio of the image and avoid distortion, 
# while also ensuring that all images have the same size, then ResizeWithPadOrCropd is a good option. 

#------------------------------------------------------------------------------
check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, num_workers=2)
check_data = monai.utils.misc.first(check_loader)
# print('sanity check:', check_data["img"].shape, check_data["label"].shape)check_data = monai.utils.misc.first(check_loader)
print('sanity check:\n', check_data["img"].shape, torch.max(check_data["img"]), "\n", check_data["label"].shape, torch.max(check_data["label"]))
print("========"*10)
check_img= check_data["img"][0, 0, :, :]
check_mask=check_data["label"][0, 0, :, :]
# Plot the image and mask
fig, ax = plt.subplots(1, 3)
ax[0].imshow(check_img, cmap="gray")
ax[0].set_title("Image")
ax[1].imshow(check_mask, cmap="gray")
ax[1].set_title("Mask")

ax[2].imshow(check_img, cmap="gray")
ax[2].imshow(check_mask, cmap="gray", alpha= 0.3)
ax[2].set_title("Mask overlay")
plt.show()
plt.show()
#------------------------------------------------------------------------------
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available() )

val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
#------------------------------------------------------------------------------
# Get a batch of data
batch = next(iter(train_loader))

# Get the first image and mask from the batch
img = batch["img"][0][0]
mask = batch["label"][0, 0]
print(img.shape)
# Plot the image and mask
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Image")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Mask")

ax[2].imshow(img, cmap="gray")
ax[2].imshow(mask, cmap="gray", alpha= 0.3)
ax[2].set_title("Mask overlay")
plt.show()
#------------------------------------------------------------------------------
roi_size= img_size
num_classes= 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= SwinUNETR(
                        img_size=roi_size, 
                        in_channels=1,  ## 3 channels, R,G,B, 1 channels for Grayscale
                        out_channels=num_classes,
                        feature_size=24, # should be divisible by 12
                        spatial_dims=2
                    ).to(device)

#------------------------------------------------------------------------------
loss_function = TverskyLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
iou_metric= MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
#------------------------------------------------------------------------------
model_path = "trained_model/"
os.makedirs(model_path, exist_ok=True)
#------------------------------------------------------------------------------
max_epochs = 300
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
post_label = Compose([AsDiscrete(to_onehot=num_classes)])

for epoch in range(max_epochs):
    print("--" * 50)
#     print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["img"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
#         print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1}/{max_epochs} - average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["img"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = roi_size
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_path, "best_metric_model.pth"))
#                 print("saved new best metric model")
            print(f"Current mean dice: {metric:.4f} | best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
#------------------------------------------------------------------------------
print(f"train completed, best_metric: {best_metric:.4f} "f"at epoch: {best_metric_epoch}")
#--------------------------------------------------------------------plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
#------------------------------------------------------------------------------
test_transforms_encoded = Compose([
    LoadImaged(keys=["img", "label"], reader=PILReader),
    EnsureChannelFirstd(keys=["img", "label"]),
    ScaleIntensityd(keys=["img"], allow_missing_keys=True),
    ResizeWithPadOrCropd(keys=["img", "label"], spatial_size=img_size),
    Flipd(keys=["img", "label"], spatial_axis=0),
    Rotate90d(keys=["img", "label"], k=3),    
    EnsureTyped(keys=["img", "label"]),

#     Resized(keys=["img", "label"], spatial_size=img_size, mode="bilinear"), 
#     SpatialPadd(keys=["img", "label"], spatial_size=1024),
    AsDiscreted(keys=["label"], to_onehot=num_classes),    
])

test_ds_encoded = monai.data.Dataset(data=test_files, transform=test_transforms_encoded)
test_loader_encoded = DataLoader(test_ds_encoded, batch_size=1, shuffle=False, num_workers=1)
#------------------------------------------------------------------------------
out_dir = "predicted_dir/"
os.makedirs(out_dir, exist_ok=True)
#------------------------------------------------------------------------------
# Get a batch of data
batch = next(iter(test_loader_encoded))

# Get the first image and mask from the batch
img = batch["img"][0][0]
mask = batch["label"][0, 0]

# Plot the image and mask
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Image")
ax[1].imshow(mask, cmap="gray")
ax[1].set_title("Mask")
plt.show()
#------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(os.path.join(model_path, "best_metric_model.pth")))
model = model.to(device) 

post_trans = Compose([Activations(softmax=True)])
model.eval()
with torch.no_grad():
    dice_scores_classes = []
    iou_scores_classes = []
    for test_data in test_loader_encoded:
        test_images, test_labels = test_data["img"].to(device), test_data["label"].to(device)
        roi_size = roi_size
        sw_batch_size = 4
        test_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        test_labels = decollate_batch(test_labels)
        dice_metric(y_pred=test_outputs, y=test_labels)
        dice_score= dice_metric(y_pred=test_outputs, y=test_labels)
        iou_score = iou_metric(y_pred=test_outputs, y=test_labels)
        dice_avg = dice_score.mean().item()
        iou_avg = iou_score.mean().item()
        dice_classes= dice_score.tolist()
        iou_classes= iou_score.tolist()
        dice_report= [dice_classes, dice_avg]
        iou_report= [iou_classes, iou_avg]
        dice_scores_classes.append(dice_report)
        iou_scores_classes.append(iou_report)
dice_scores_classes = np.array(dice_scores_classes, dtype=object)
mean_avg_dice_score = dice_scores_classes[:, 1].mean()

iou_scores_classes = np.array(iou_scores_classes, dtype=object)
mean_avg_iou_score = iou_scores_classes[:, 1].mean()
print(f"Dice score on test set:  {mean_avg_dice_score*100:.6}%, IoU score on test set:  {mean_avg_iou_score*100:.6}%")
#------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normalize = ScaleIntensity()
post_trans = Compose([Activations(softmax=True)])

with torch.no_grad():        
    for test_data in test_loader_encoded:
        test_inputs = test_data["img"].to(device)
        test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_outputs, test_labels = from_engine(["pred", "label"])(test_data)
        test_outputs = test_outputs.to(device)  # move the tensors to the same device
        test_labels = test_labels.to(device)
        test_outputs = [post_trans(i) for i in decollate_batch(test_outputs)]
        test_labels = decollate_batch(test_labels)       
        dice_score = dice_metric(y_pred=test_outputs, y=test_labels)
        iou_score = iou_metric(y_pred=test_outputs, y=test_labels)
        predicted = torch.argmax(test_data["pred"], dim=1, keepdim=True).detach().cpu()
        predicted_mask= np.moveaxis(np.squeeze(predicted, 0), 0, -1)
        filename = os.path.basename(test_data["img_meta_dict"]["filename_or_obj"][0])
        predicted_filename = os.path.join(out_dir, f"predicted_{filename}")
        print(f"{filename} ==> dice score:  {dice_score.mean().item()}, iou metric: {iou_score.mean().item()}")
        predicted_mask_array = array_to_img(predicted_mask)
        predicted_mask_array.save(os.path.join(out_dir, filename))
        
        plt.figure("check", (18, 6))
        
        plt.subplot(1, 5, 1)
        plt.title("test image")
#         plt.title(f"{filename}")
        in_img = test_data["img"][0, 0, :, :].detach().cpu()
        plt.imshow(in_img, cmap="gray")
        
        plt.subplot(1, 5, 2)
        plt.title("ground truth")
        test_truth = test_data["label"]
        print(test_truth.shape)
        plt.imshow(np.argmax(test_truth.detach().cpu().numpy(), axis=1)[0,1: :, :],cmap="gray")
        
        plt.subplot(1, 5, 3)
        plt.title("predicted mask")
        plt.imshow(predicted_mask_array,cmap="gray")
        
        plt.subplot(1, 5, 4)
        plt.title("ground truth overlay")
        plt.imshow(in_img, cmap= 'gray')
        plt.imshow(np.argmax(test_truth.detach().cpu().numpy(), axis=1)[0,1: :, :],cmap="gray", alpha=0.3)
        
        plt.subplot("predicted mask overlay")
        plt.imshow(in_img, cmap= 'gray')
        plt.imshow(predicted_mask_array, cmap="gray", alpha= 0.3) 
        
        plt.show()
        print("=========" * 10)

#------------------------------------------------------------------------------
print(train_files)
print("=====================================================================================\n")
print(val_files)
print("=====================================================================================\n")
print(test_files)
#------------------------------------------------------------------------------
