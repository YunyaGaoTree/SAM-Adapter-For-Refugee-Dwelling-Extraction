# Data processing for satellite image data for Segment Anything Model
# Author: Yunya Gao
# Date: 06,December,2023

import os
import numpy as np
import json
from tqdm import tqdm
from itertools import groupby
import shutil
import glob
import math
import subprocess
import sys
import scipy
import pathlib
from pathlib import Path

# import warnings
# # stop popping up warning
# warnings.filterwarnings("ignore", message="global net_impl.cpp:178 setUpNet DNN module was not built with CUDA backend; switching to CPU")
import sys

# Redirect stderr to /dev/null or NUL to suppress warnings
if os.name == 'posix':  # On Linux/Unix/Mac
    sys.stderr = open(os.devnull, 'w')
elif os.name == 'nt':  # On Windows
    sys.stderr = open(os.devnull, 'w')

import cv2
from skimage import io
from osgeo import gdal
import rasterio
from rasterio.enums import Resampling
from PIL import Image

model_name = "SAM"

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Crop large image/gt data (tiff) into smaller patches (png) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

def crop2patch(image, p_size, p_overlap):

    # p_size and p_overlap are numeric values.
    # aug_idx
    # Define patch size and overlap in all three dimensions
    if len(image.shape) == 3:
        height, width, channels = image.shape
        patch_size = (p_size, p_size, channels)  
        overlap = (p_overlap, p_overlap, 0) 
    if len(image.shape) == 2:
        height, width = image.shape
        patch_size = (p_size, p_size)  
        overlap = (p_overlap, p_overlap) 

    # Calculate the number of patches in all three dimensions
    num_patches_x = (width - patch_size[1]) // (patch_size[1] - overlap[1]) + 1
    num_patches_y = (height - patch_size[0]) // (patch_size[0] - overlap[0]) + 1
    
    # Initialize a list to store the patches
    patches = []
    
    # Extract patches with overlap
    for y in range(0, height - patch_size[0] + 1, patch_size[0] - overlap[0]):
        for x in range(0, width - patch_size[1] + 1, patch_size[1] - overlap[1]):
            patch = image[y:y + patch_size[0], x:x + patch_size[1]]
            patches.append(patch)

    return patches

    
def swap_axes_img(array):
    # this function is applied when using gdal 
    # convert shape from (channel, width, height) to (width, height, channels)
    arr1 = np.swapaxes(array, 0, -1)
    arr2 = np.swapaxes(arr1, 0, 1)

    return arr2


def contrast98_2_convert2uint8(image_data):
    
    minval = np.percentile(image_data, 2)
    maxval = np.percentile(image_data, 98)
    pixvals = np.clip(image_data, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 255
    pixvals = pixvals.astype(np.uint8)
    
    return pixvals


def data_process_sam_seg(path_dataset, data_type, patch_size, model_type):
    # data_type: train or test

    input_image_path = path_dataset+'/raw/'+data_type+'/images/'
    input_gt_path = path_dataset+'/raw/'+data_type+'/gt/'
    save_path = path_dataset+'/'+model_type+'/'+str(patch_size)+'/'+data_type+'/' 

    output_im_train = os.path.join(save_path, 'images')
    output_gt_train = os.path.join(save_path, 'gt')
    
    if not os.path.exists(output_im_train):
        os.makedirs(output_im_train)
        os.makedirs(output_gt_train)
        
    # read the name of image/label data
    input_label = os.listdir(input_gt_path)
    cp = '.ipynb_checkpoints'
    if cp in input_label:
        input_label.remove(cp)

    # set up patch size
    if patch_size > 1000:
            patch_overlap = int(7 * patch_size / 8)
    else:
            patch_overlap = int(1 * patch_size / 4)
    
    for g_id, label in enumerate(tqdm(input_label)):
        
        # read data
        label_info = [''.join(list(g)) for k, g in groupby(label, key=lambda x: x.isdigit())]
        label_name = label_info[0] + label_info[1]
        
        path_img = os.path.join(input_image_path, label_name + '.tif')
              
        # open tiff data by skimage, it does not work in colab
        # image_data = io.imread(path_img)
        # open tiff data by gdal, in colab
        image_data = gdal.Open(path_img).ReadAsArray()
        image_data = swap_axes_img(image_data)
        image_data = contrast98_2_convert2uint8(image_data)
        
        path_gt = os.path.join(input_gt_path, label_name + '.tif')
        # gt_im_data = io.imread(path_gt) 
        gt_im_data = gdal.Open(path_gt).ReadAsArray()

        gt_im_data[0, :] = 0 # because some of images with a boundary with wrong values
        gt_im_data[:, 0] = 0
        gt_im_data[:, -1] = 0
        gt_im_data[-1, :] = 0
        gt_im_data = gt_im_data.astype("uint8")
        
        im_h, im_w, _ = image_data.shape
            
        patch_img_list = crop2patch(image_data, patch_size, patch_overlap)
        patch_gt_list = crop2patch(gt_im_data, patch_size, patch_overlap)
        train_im_id = 0

        for i in range(len(patch_gt_list)):

            p_gt = patch_gt_list[i]
            p_im = patch_img_list[i]

            p_gt = p_gt.astype('uint8')
            p_gt[p_gt > 0] = 255
          
            # define patch name
            # it is important to use png, otherwise the value of input array will be chanaged
            p_name = label_name + '-' + str(train_im_id) + '.png' 
            
            # save patch image
            io.imsave(os.path.join(output_im_train, p_name), p_im, check_contrast=False)
            
            # save patch GT
            if p_gt.shape[-1] == 1:
                p_gt_ = np.expand_dims(p_gt, axis=-1)
                p_gt_3ch = np.concatenate((p_gt_, p_gt_, p_gt_), axis=-1)
            else:
                p_gt_3ch = p_gt
                
            io.imsave(os.path.join(output_gt_train, p_name), p_gt_3ch, check_contrast=False)
            
            train_im_id += 1


def data_process_sam_seg_final(path_database, data_list, type_list, patch_size):
    for dataset in data_list:
        for dtype in type_list:
            
            path_dataset = os.path.join(path_database, dataset)
            print("Start processing: " + dataset + " " + dtype + " " + str(patch_size))

            data_process_sam_seg(path_dataset, dtype, patch_size, model_name)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Data augmentation: Flip, Rotation 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -               

def copy_data_for_augmentation(source_directory, destination_directory):

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # List files in the source directory
    files_to_copy = os.listdir(source_directory)
    
    # Copy each file from the source directory to the destination directory
    for file_name in files_to_copy:
        source_file_path = os.path.join(source_directory, file_name)
        destination_file_path = os.path.join(destination_directory, file_name)
        shutil.copy2(source_file_path, destination_file_path)

    file_list = glob.glob(source_directory + "/*.png")
    
    return file_list


def manipulate_image(input_path, output_path, operation="rotate", degrees=90):
    # Load the image
    img = Image.open(input_path)

    # Perform the specified operation
    if operation == "rotate":
        modified_img = img.rotate(degrees) 
    elif operation == "horizontal_flip":
        modified_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif operation == "vertical_flip":
        modified_img = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError("Operation can only be 'rotate', 'horizontal_flip', or 'vertical_flip'")

    # Save the modified image as PNG
    modified_img.save(output_path, "PNG")
    

def augmentation_single(data_list, dtype, operation, degrees, op_times, path_base):
    
    num_base = len(data_list) * op_times
    
    for input_path in data_list:

        filename = os.path.basename(input_path)
        idx = int(filename.split('-')[-1].split('.')[0])
        name = filename.split('-')[0]
        
        output_path = os.path.join(path_base, name+"-"+str(idx+num_base)+".png")
        manipulate_image(input_path, output_path, operation, degrees)
        
    
def data_process_augmentation(path_base_img, img_list, path_base_gt, gt_list, dtype, operation, degrees, op_times):
    # op_times: which time of operatios, 1st time, 2nd time...
    
    augmentation_single(img_list, dtype, operation, degrees, op_times, path_base_img)
    augmentation_single(gt_list, dtype, operation, degrees, op_times, path_base_gt)


def data_process_augmentation_final(path_database, data_list, type_list, patch_size, operation_list, degrees_list, aug_idx=""):

    for dataset in data_list:
        for dtype in type_list:

            path_base = os.path.join(path_database, dataset, model_name, str(patch_size))
            print("Start processing: "+path_base)

            # copy image data
            src_img = os.path.join(path_base, dtype, "images")
            aug_img = os.path.join(path_base, dtype+"_aug"+str(aug_idx), "images")
            img_list = copy_data_for_augmentation(src_img, aug_img)
          
            # copy gt data
            src_gt = os.path.join(path_base, dtype, "gt")
            aug_gt = os.path.join(path_base, dtype+"_aug"+str(aug_idx), "gt")
            gt_list = copy_data_for_augmentation(src_gt, aug_gt)

            op_times = 1

            for operation in operation_list:
                
                if operation != "rotate":
                    
                    degrees = 0 # no use here
                    # path_base_img, img_list, path_base_gt, gt_list, dtype, operation, degrees, op_times
                    data_process_augmentation(aug_img, img_list, aug_gt, gt_list, dtype, operation, degrees, op_times)
                    op_times += 1
                    
                elif operation == "rotate":
                    
                    for degrees in degrees_list:    
                        data_process_augmentation(src_img, img_list, src_gt, gt_list, dtype, operation, degrees, op_times)
                        op_times += 1
                        
                else:
                    print(operation + " is not included in the codes. Please try horizontal_flip, vertical_flip, rotate")

        print("Done.")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Super resolution model for training data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
                
import cv2
    
def set_sr_model():
    # if cannot import dnn_superres from cv2, then: 
    # pip install --upgrade opencv-python
    # pip install --upgrade opencv-contrib-python

    # links to download models
    # https://github.com/Saafke/EDSR_Tensorflow/blob/master/models/
    # https://github.com/fannymonori/TF-LapSRN/blob/master/export/LapSRN_x8.pb
    # feel free to use other SR models

    from cv2 import dnn_superres

    # initialize super resolution object
    sr = dnn_superres.DnnSuperResImpl_create()

    # read the model
    path = 'EDSR_x4.pb'
    sr.readModel(path)

    # set the model and scale
    sr.setModel('edsr', 4)

    # if you have cuda support
    sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    return sr

def upscale_img_sr(path_in, path_out, sr_model):
    
    # load the image
    image = cv2.imread(path_in)

    # upsample the image
    upscaled = sr_model.upsample(image)

    # save the upscaled image
    cv2.imwrite(path_out, upscaled)

  
def upscale_img_cubic(path_in, path_out):
    
    # load the image
    img = cv2.imread(path_in)

    # upsample the image
    width = 1024
    height = 1024
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC) # INTER_NEAREST INTER_LINEAR
    
    # save the upscaled image
    cv2.imwrite(path_out, resized)


def upscale_lab(path_in, path_out):
    
    # load the image
    lab = cv2.imread(path_in)

    # upsample the image
    upscale_times = 4 # ratio of original size
    width = int(lab.shape[1] * upscale_times)
    height = int(lab.shape[0] * upscale_times)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(lab, dim, interpolation=cv2.INTER_LINEAR)
    resized[resized>0] = 255

    # save the upscaled image
    cv2.imwrite(path_out, resized)


def upscale_data_by_SR_final(path_database, data_sr_list, type_sr_list):

    # for SR model, the only currently supported patch size is 256.
    patch_size = 256
    upscaled_folder = "upscaled"

    for dataset in data_sr_list:
        for dtype in type_sr_list:

            path_dataset = os.path.join(path_database, dataset, model_name, str(patch_size), dtype, "images")
            img_name_list = os.listdir(path_dataset)
            print("Start processing: {}    {}".format(dataset, dtype))

            for img_name in img_name_list:

                path_ups_img = os.path.join(path_database, dataset, model_name, upscaled_folder, "SR", dtype, "images")
                Path(path_ups_img).mkdir(parents=True, exist_ok=True)

                path_ups_lab = os.path.join(path_database, dataset, model_name, upscaled_folder, "SR", dtype, "gt")
                Path(path_ups_lab).mkdir(parents=True, exist_ok=True)

                path_in_img = os.path.join(path_database, dataset, model_name, str(patch_size), dtype, "images", img_name)
                path_out_img = os.path.join(path_ups_img, img_name)
                path_in_lab = os.path.join(path_database, dataset, model_name, str(patch_size), dtype, "gt", img_name)
                path_out_lab = os.path.join(path_ups_lab, img_name)

                # set up sr model
                sr_model = set_sr_model()

                # upscale images
                upscale_img_sr(path_in_img, path_out_img, sr_model)

                # upscale labels
                upscale_lab(path_in_lab, path_out_lab)
    print("Done.")


def upscale_data_by_cubic_final(path_database, data_sr_list, type_sr_list):

    # for SR model, the only currently supported patch size is 256.
    patch_size = 256
    upscaled_folder = "upscaled"

    for dataset in data_sr_list:
        for dtype in type_sr_list:

            path_dataset = os.path.join(path_database, dataset, model_name, str(patch_size), dtype, "images")
            img_name_list = os.listdir(path_dataset)
            print("Start processing: {}    {}".format(dataset, dtype))

            for img_name in img_name_list:

                path_ups_img = os.path.join(path_database, dataset, model_name, upscaled_folder, "cubic", dtype, "images")
                Path(path_ups_img).mkdir(parents=True, exist_ok=True)

                path_ups_lab = os.path.join(path_database, dataset, model_name, upscaled_folder, "cubic", dtype, "gt")
                Path(path_ups_lab).mkdir(parents=True, exist_ok=True)

                path_in_img = os.path.join(path_database, dataset, model_name, str(patch_size), dtype, "images", img_name)
                path_out_img = os.path.join(path_ups_img, img_name)
                path_in_lab = os.path.join(path_database, dataset, model_name, str(patch_size), dtype, "gt", img_name)
                path_out_lab = os.path.join(path_ups_lab, img_name)

                # upscale images
                upscale_img_cubic(path_in_img, path_out_img)

                # upscale labels
                upscale_lab(path_in_lab, path_out_lab)
    print("Done.")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Super resolution model for testing data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

def upscale_testing_image_by_SR(path_in_img, sr_model, factor=4, patch_size=512):
    
    # set patch_size as 512 rather than 1024 to have some overlap area to produce seamless output image
    
    # Step 0: Set up basic information
    input_array = cv2.imread(path_in_img)
    
    h_img = input_array.shape[0]
    w_img = input_array.shape[1]
    n_ch = input_array.shape[2]

    # Step 1: Determine the dimensions of the new array after upscaling
    new_shape = (h_img * factor, w_img * factor, n_ch)

    # Step 2: Pad the input array to make it divisible by the patch size
    num_patches_height = h_img // patch_size + 1
    num_patches_width = w_img // patch_size + 1

    print("num_patches_height: {}, num_patches_width: {}".format(num_patches_height, num_patches_width))

    # Step 3: Process and upscale overlapping patches
    new_array = np.zeros(new_shape)

    for h_idx in range(num_patches_height):
        for w_idx in range(num_patches_width):

            # Calculate the starting and ending indices of the patch
            start_h = h_idx * patch_size
            start_w = w_idx * patch_size

            if start_h + patch_size < h_img or start_h + patch_size == h_img:
                end_h = start_h + patch_size
            else:
                start_h = h_img - patch_size
                end_h = h_img

            if start_w + patch_size < w_img or start_w + patch_size == w_img:
                end_w = start_w + patch_size
            else:
                start_w = w_img - patch_size
                end_w = w_img

            # Extract the patch from the input array
            patch = input_array[start_h:end_h, start_w:end_w, :]

            # Process the patch (you can replace this line with any operation you want)
            upscaled_patch = sr_model.upsample(patch)

            # Append the upscaled patch to the new array
            start_h_new = h_idx * patch_size * factor
            end_h_new = (h_idx + 1) * patch_size * factor
            h_last = -1 * patch_size * factor

            start_w_new = w_idx * patch_size * factor
            end_w_new = (w_idx + 1) * patch_size * factor
            w_last = -1 * patch_size * factor

            if h_idx < num_patches_height-1: 

                if w_idx < num_patches_width-1:
                    new_array[start_h_new:end_h_new, start_w_new:end_w_new, :] = upscaled_patch
                else:
                    new_array[start_h_new:end_h_new, w_last:, :] = upscaled_patch

            else:
                if w_idx < num_patches_width-1:
                    new_array[h_last:, start_w_new:end_w_new, :] = upscaled_patch
                else:
                    new_array[h_last:, w_last:, :] = upscaled_patch

        print("Done:    Row: {}".format(h_idx))

    new_array = new_array.astype("uint8")
    
    return new_array


def upscale_testing_image_by_SR_final(path_database, dataset, data_type):
    
    dtype = "test"
    upscale_factor = 4
    upscaled_folder = "SR"

    path_test_img = os.path.join(path_database, dataset, "raw", "test", data_type)
    path_list = os.listdir(path_test_img)

    in_img_list = []
    for filename in path_list:
        if filename[-4:] == ".tif":
            in_img_list.append(filename)

    in_img_list.sort()

    path_out = os.path.join(path_database, dataset, model_name, "upscaled", dtype, upscaled_folder, data_type)
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    for in_img in in_img_list:
    
        path_out_img = os.path.join(path_out, in_img)
        path_in_img = os.path.join(path_test_img, in_img)
        print("Processing by {}: {}".format(upscaled_folder, path_in_img))
        
        sr_model = set_sr_model()
        
        upscaled_img = upscale_testing_image_by_SR(path_in_img, sr_model)
        
        b1 = upscaled_img[...,:1]
        b2 = upscaled_img[...,1:2]
        b3 = upscaled_img[...,2:]
        upscaled_img = np.concatenate((b3, b2, b1), axis=-1)
        
        upscaled_img_axis = np.swapaxes(upscaled_img, 2, 0)
        upscaled_img_axis = np.swapaxes(upscaled_img_axis, 1, 2)
        
        with rasterio.open(path_in_img) as dataset:
    
            # resample data to target shape using upscale_factor
            height = int(dataset.height * upscale_factor)
            width = int(dataset.width * upscale_factor)
    
            # scale image transform
            dst_transform = dataset.transform * dataset.transform.scale(
                (dataset.width / upscaled_img_axis.shape[-1]),
                (dataset.height / upscaled_img_axis.shape[-2])
            )
    
            # Write outputs
            # set properties for output
            dst_kwargs = dataset.meta.copy()
            dst_kwargs.update(
                {
                    "crs": dataset.crs,
                    "transform": dst_transform,
                    "width": upscaled_img_axis.shape[-1],
                    "height": upscaled_img_axis.shape[-2],
                    "nodata": 0,  
                }
            )
    
            with rasterio.open(path_out_img, "w", **dst_kwargs) as dst:
                # iterate through bands
                for i in range(upscaled_img_axis.shape[0]):
                      dst.write(upscaled_img_axis[i].astype(rasterio.uint8), i+1)


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Upscaling for testing data by cubic interpretation
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

def upscale_testing_data_cubic_final(path_database, dataset, data_type, patch_size):
    
    dtype = "test"
    upscaled_folder = 'cubic'
    upscale_factor = 1024 / patch_size

    resampling = Resampling.cubic
    # resampling = Resampling.bilinear
    # resampling = Resampling.nearest
   
    path_test_img = os.path.join(path_database, dataset, "raw", "test", data_type)
    path_list = os.listdir(path_test_img)

    in_img_list = []
    
    for filename in path_list:
        if filename[-4:] == ".tif":
            in_img_list.append(filename)
            
    in_img_list.sort()

    if data_type == "images":
        path_out= os.path.join(path_database, dataset, model_name, "upscaled", dtype, upscaled_folder, data_type)
        
    elif data_type == "gt":
        path_out= os.path.join(path_database, dataset, model_name, "upscaled", dtype, upscaled_folder, data_type)
        
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

    for in_img in in_img_list:
    
        path_out_img = os.path.join(path_out, in_img)
        path_in_img = os.path.join(path_test_img, in_img)
        print("Processing by {}: {}".format(upscaled_folder, path_in_img))
    
        with rasterio.open(path_in_img) as dataset:
    
            # resample data to target shape using upscale_factor
            height = int(dataset.height * upscale_factor)
            width = int(dataset.width * upscale_factor)
    
            # the virtual of data is a numpy array in the shape of (n-band, height, width)
            # so, replace this part with the output from SR model
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    height,
                    width
                ),
                resampling=resampling
            )
            print(data.shape)
    
            # scale image transform
            dst_transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
            )
    
            # Write outputs
            # set properties for output
            dst_kwargs = dataset.meta.copy()
            dst_kwargs.update(
                {
                    "crs": dataset.crs,
                    "transform": dst_transform,
                    "width": data.shape[-1],
                    "height": data.shape[-2],
                    "nodata": 0,  
                }
            )
    
            with rasterio.open(path_out_img, "w", **dst_kwargs) as dst:
                # iterate through bands
                for i in range(data.shape[0]):
                      dst.write(data[i].astype(rasterio.uint8), i+1)