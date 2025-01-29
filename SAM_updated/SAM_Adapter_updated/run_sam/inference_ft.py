import sys
import os

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import argparse
import numpy as np
import yaml
from tqdm import tqdm
import cv2
import glob
import scipy
import imagecodecs
from skimage.transform import resize
from skimage import io 

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.distributed as dist
from torchvision.transforms import functional as F

import datasets
import models
import utils

from common_ft import *
import pathlib
import geopandas as gpd
import rasterio

import matplotlib.pyplot as plt

# torch.distributed.init_process_group(backend='nccl')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_yaml(data_yaml, path_out):
    """ A function to write YAML file"""
    
    path_out_yaml = os.path.join(path_out, "config.yaml")
    with open(path_out_yaml, 'w') as f:
        yaml.dump(data_yaml, f)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ann=None):
        if ann is None:
            for t in self.transforms:
                image = t(image)
            return image
        for t in self.transforms:
            image, ann = t(image, ann)
        return image, ann


class Resize(object):
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width = image_width
        self.ann_height = ann_height
        self.ann_width = ann_width

    def __call__(self, image, ann):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0

        sx = self.ann_width / ann['width']
        sy = self.ann_height / ann['height']
        ann['junc_ori'] = ann['junctions'].copy()
        ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0, self.ann_width - 1e-4)
        ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0, self.ann_height - 1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height
        ann['mask_ori'] = ann['mask'].copy()
        ann['mask'] = cv2.resize(ann['mask'].astype(np.uint8), (int(self.ann_width), int(self.ann_height)))

        return image, ann


class ResizeImage(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image, ann=None):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0
        if ann is None:
            return image
        return image, ann


class ToTensor(object):
    def __call__(self, image, anns=None):
        if anns is None:
            return F.to_tensor(image)

        for key, val in anns.items():
            if isinstance(val, np.ndarray):
                anns[key] = torch.from_numpy(val)
        return F.to_tensor(image), anns
    

def inference_image(image, model):
    
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
                ])

    h_stride, w_stride = 600, 600
    h_crop, w_crop = 1024, 1024
    h_img, w_img, _ = image.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
    count_mat = np.zeros([h_img, w_img])
    juncs_whole_img = []

    patch_weight = np.ones((h_crop + 2, w_crop + 2))
    patch_weight[0,:] = 0
    patch_weight[-1,:] = 0
    patch_weight[:,0] = 0
    patch_weight[:,-1] = 0

    patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)
    patch_weight = patch_weight[1:-1,1:-1]

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop_img = image[y1:y2, x1:x2, :]
            crop_img = crop_img.astype(np.float32)
            crop_img_tensor = transform(crop_img)[None].to(device)

            meta = {
                'height': crop_img.shape[0],
                'width': crop_img.shape[1],
                'pos': [x1, y1, x2, y2]
            }

            # get the sum of an image tensor
            tensor_sum = crop_img_tensor.sum().item()

            # if the image is an empty tensor
            if tensor_sum < 10: 
                # the output mask_pred is also empty
                mask_pred = torch.zeros_like(crop_img_tensor).cpu().numpy().copy()[0,0]

            else:
                with torch.no_grad():
                    mask_pred = model.infer(crop_img_tensor)
                    mask_pred = torch.sigmoid(mask_pred)
                    mask_pred = mask_pred.cpu().numpy().copy()[0,0]

            mask_pred *= patch_weight
            pred_whole_img += np.pad(mask_pred,
                                ((int(y1), int(pred_whole_img.shape[0] - y2)),
                                (int(x1), int(pred_whole_img.shape[1] - x2))))
            count_mat[y1:y2, x1:x2] += patch_weight

    pred_whole_img = pred_whole_img / count_mat

    return pred_whole_img


def get_binary_mask(pred_mask, thres): 
    
    binar_mask = pred_mask > thres
    
    return binar_mask


def save_fig(data, data_name, path_out):

    # create figure
    plt.figure(figsize=(50, 50))
    plt.axis('off')
    
    # save data
    path_out_ = os.path.join(path_out, data_name+".png")
    plt.imshow(data)
    plt.savefig(path_out_, bbox_inches='tight')
    

def tiff_to_shp(tiff_path, output, simplify_tolerance=0.001, **kwargs):
    """Convert a tiff file to a shapefile.
    Args:
        tiff_path (str): The path to the tiff file.
        output (str): The path to the shapefile.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """
    raster_to_shp(tiff_path, output, simplify_tolerance=simplify_tolerance, **kwargs)


def save_predicted_probability_mask_shapefile(img_path, pred_whole_img, thres, path_out, upsample, area_idx):

    # read spatial information from test image
    with rasterio.open(img_path) as src:
        ras_meta = src.profile
        crs = src.crs # for vector
        ras_meta["count"] = 1 # for raster
        ras_meta["dtype"] = "float32"

    pred_whole_img_prob = np.expand_dims(pred_whole_img[...], axis=0)
    pred_whole_img_bin = pred_whole_img_prob > thres
    
    # save probability as png
    save_fig(pred_whole_img, "mask_prob", path_out)
    # save binary dense mask as png
    save_fig(pred_whole_img_bin, "mask_binary", path_out)

    # save probability map
    if upsample != "SR":
        path_mask_out = path_out+"/pred_mask_prob.tif"
        with rasterio.open(path_mask_out, 'w', **ras_meta) as dst:
            dst.write(pred_whole_img_prob)

    # save predicted binary mask - stop saving them to save time and space
    path_mask_out_bin = path_out+"/pred_mask_bin"+str(thres)+".tif"
    with rasterio.open(path_mask_out_bin, 'w', **ras_meta) as dst:
        dst.write(pred_whole_img_bin)

    # save polygons as shapefile if the polgyons are not None.
    num_positive = np.sum(pred_whole_img_bin)
    num_total = pred_whole_img_bin.shape[1] * pred_whole_img_bin.shape[2]
    
    if num_positive > num_total * 1e-4:
        path_shp_out = path_out+"/pred_mask_bin"+str(thres)+".shp"
        tiff_to_shp(path_mask_out_bin, path_shp_out)

    
# main function
def inference_main(model, config, thres, path_out, upsample, path_img_list, path_gt_list):

    for i in range(len(path_img_list)):
    
        # read image for testing (a large geotiff data)
        img_path = path_img_list[i]
        image = io.imread(img_path)
        image = (image - image.min()) / (image.max() - image.min())
        
        # read ground truth (gt) data 
        gt_path = path_gt_list[i]
        gt = io.imread(gt_path)
    
        # probability map
        prob_mask = inference_image(image, model)
        print("Predicted probability map.")
        
        # save probability map, binary mask, shapefile
        area_idx = "area"+str(i+1)
        path_out_final = os.path.join(path_out, area_idx)
        pathlib.Path(path_out_final).mkdir(parents=True, exist_ok=True)
        
        save_predicted_probability_mask_shapefile(img_path, prob_mask, thres, path_out_final, upsample, area_idx)
        print("Save predicted probability map, binary map with threshold at {} and shapefile.".format(thres))

        # save image and gt
        save_fig(image, "image", path_out_final)
        save_fig(gt, "gt", path_out_final)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/config_sam_vit_h.yaml", help="use the hyperparameters provided by SAM-Adapter")
    parser.add_argument('--data', default=None, help="different datasets")
    parser.add_argument('--upsample', default="1024", help="1024 or upscaled") 
    parser.add_argument('--size', default="small", help="small or large") 
    parser.add_argument('--uptype', default="cubic", help="cubic or SR") 
    parser.add_argument('--model_save_epoch', default=2, help="the epoch of pretrained model") 
    parser.add_argument('--thres', default=0.5, help="the threshold to determine the binary map") 
    
    # path_data = "path of your data folder" # e.g. "/home/usr/Data"
    path_data = "/home/ubuntu/SAM1/Data" 
    
    args = parser.parse_args()
    path_cfg = args.config
    data_name = args.data
    size = args.size
    upsample = args.upsample
    uptype = args.uptype
    model_save_epoch = args.model_save_epoch
    thres = args.thres
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # open the base config file to read hyperparameters
    with open(path_cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
   # testing data
    if upsample == "1024":
        path_test_data = os.path.join(path_data, data_name, "raw", "test")
        path_img_list = glob.glob(os.path.join(path_test_data, "images") + "/*.tif")
        
    elif upsample == "upscaled":
        path_test_data = os.path.join(path_data, data_name, "SAM", upsample, "test")
        path_img_list = glob.glob(os.path.join(path_test_data, uptype, "images") + "/*.tif")
    
    path_gt_list = glob.glob(os.path.join(path_test_data, "cubic", "gt") + "/*.tif")

    path_img_list.sort()
    path_gt_list.sort()

    # defin path of outputs of predicted results to be saved
    path_output = os.path.join('outputs_pretrained', data_name, size, upsample, uptype, "epoch"+str(model_save_epoch))
    pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    
    write_yaml(config, path_output)
    print('config saved.')

    path_model_save = os.path.join('save_model', data_name, size, upsample, uptype)
    path_model_pretrained = os.path.join(path_model_save, "model_epoch"+str(model_save_epoch)+".pth")
    pretrained_dict = torch.load(path_model_pretrained, map_location=torch.device('cuda'), weights_only=True)
    
    model = models.make(config['model']).cuda()
    model.load_state_dict(pretrained_dict)
    
    with torch.no_grad():
        inference_main(model, config, thres, path_output, upsample, path_img_list, path_gt_list)