import sys
import os

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import argparse
import leafmap
import yaml
import pathlib
import glob

from samgeo.samgeo import SamGeo
from samgeo.common import get_basemaps

import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch.nn as nn
import torch.nn.functional as F

    
# main function
def main(path_model_checkpoint, img_path, mask_path, shp_path):
    
    # load pretrained model from meta: sam_vit_h_4b8939
    sam = SamGeo(
        model_type="vit_h",
        checkpoint=path_model_checkpoint,
        sam_kwargs=None)
    
    # create mask
    sam.generate(img_path, mask_path, batch=True, foreground=True, erosion_kernel=(3, 3), mask_multiplier=255)
    
    # from mask to polygons
    sam.tiff_to_vector(mask_path, shp_path)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', default="./pretrained/sam_vit_h_4b8939.pth")
    parser.add_argument('--thres', default=0.5, help="threshold to get binary mask", type=float)
    parser.add_argument('--data', default=None, help="Dagaha2017 Djibo2019...")
    parser.add_argument('--upsample', default="1024", help="1024 or SR") 
    parser.add_argument('--uptype', default="", help="nearest bilinear EDSR") 
    
    # path_data = "path of your data folder" # e.g. "/home/usr/Data"
    path_data = "/home/yunya/anaconda3/envs/Data"
   
    args = parser.parse_args()
    thres = args.thres
    path_model_checkpoint = args.model
    data_name = args.data
    upsample = args.upsample
    uptype = args.uptype
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # read image for testing (a large geotiff data)
    if upsample == "1024":
        path_test_data = os.path.join(path_data, data_name, "raw", "test")
        path_img = glob.glob(os.path.join(path_test_data, "images") + "/*.tif")[0]
        path_output = os.path.join('outputs', data_name, "noFT", upsample)
        
    elif upsample == "SR":
        path_test_data = os.path.join(path_data, data_name, "SAM", upsample, "test")
        path_img = glob.glob(os.path.join(path_test_data, uptype) + "/*.tif")[0]
        path_output = os.path.join('outputs', data_name, "noFT", upsample, uptype)
    
    # define output predicted mask data and polygon data
    pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    print(path_output)
    
    path_mask_out = os.path.join(path_output, "pred_mask_noFT.tif")
    path_shp_out = os.path.join(path_output, "pred_mask_noFT.shp")
    
    # run main function
    main(path_model_checkpoint, path_img, path_mask_out, path_shp_out)