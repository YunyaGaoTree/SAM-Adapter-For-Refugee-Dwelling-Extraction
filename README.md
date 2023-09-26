# Segment Anything Model for Refugee-Dwelling Extraction (SAM4Refugee) From High-Resolution Satellite Imagery

### Updated 26/09/2023. This repository is still under construction.

This repository contains codes for how to segment refugee dwellings (or any other types of buildings) from high-resolution satellite imagery using the SAM-Adapter model.<br>

The codes are adapted based on [SAM Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) for training and [segment-geospatial](https://github.com/opengeos/segment-geospatial) for creating prediceted masks in the format of GeoTIFF and polygons in the format of ShapeFile.<br>

These codes can be easily adapted for binary semantic segmentation applications in remote sensing. Feel free to use it for your own applications and implement in your local machine.<br>

**Avaialable input prompts: <br>**
- ('--config', default="configs/config_sam_vit_h.yaml", help="use the hyperparameters provided by SAM-Adapter")
- ('--data', default=None, help="different datasets")
- ('--upsample', default="1024", help="1024 or SR") 
- ('--size', default="small", help="small or large") 
- ('--uptype', default="", help="nearest bilinear EDSR") 
- ('--epoch', default=15, help="epochs for training") 
- ('--model_save_epoch', default=999, help="the interval of saving trained models.") 
- ('--inference_save_epoch', default=1, help="the interval of saving trained models") 
- ('--thres', default=0.5, help="the threshold to determine the binary map")  

**Before using this repository:**
Please Change "path_data" in /run_sam/train.py & inference_noft.py & evaluation.py to your own path.

Check **"SAM_Adapter_For_Refugee_Dwellings.ipynb"** for more details to use prompts to run the SAM-Adapter for different purposes.

**Avaiable datasets:**
Due to data license issues, only data of Kutupalong refugee camp is available.
Through the following link, you could download the raw image and lable data for training (small and large), testing and validation.
https://drive.google.com/drive/folders/1cenZaqgjbvGqlk3c8Y4JDfNoIJBZCvn1?usp=sharing

Link to the complete drone image dataset for Kutupalong refugee camp:
https://data.humdata.org/dataset/iom-bangladesh-needs-and-population-monitoring-npm-drone-imagery-and-gis-package-by-camp-august-2018
