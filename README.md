# Segment Anything Model for Refugee-Dwelling Extraction (SAM4Refugee) From High-Resolution Satellite Imagery

### Updated 17/03/2024. 

This repository contains codes for how to segment refugee dwellings (or any other types of buildings) from high-resolution satellite imagery using the SAM-Adapter model.<br>

The codes are adapted based on [SAM Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) for training and [segment-geospatial](https://github.com/opengeos/segment-geospatial) for creating prediceted masks in the format of GeoTIFF and polygons in the format of ShapeFile.<br>

These codes can be easily adapted for binary semantic segmentation applications in remote sensing. Feel free to use it for your own applications and implement in your local machine.<br>

**Install environment**

0. Download data:
Due to data license issues, only data of Kutupalong refugee camp is available.
Through the following link, you could download the raw image and lable data for training (small and large), testing and validation.
https://drive.google.com/drive/folders/1FgD-E_2RwSeVwkgW2X7JwmmLnMU_Zo2z?usp=sharing

Link to the complete drone image dataset for Kutupalong refugee camp:
https://data.humdata.org/dataset/iom-bangladesh-needs-and-population-monitoring-npm-drone-imagery-and-gis-package-by-camp-august-2018

1. Change the path of data based on your own situations.
Please Change "path_data" in /run_sam/train.py & inference_noft.py & evaluation.py to your own path.

3. Try “environment.yml” at first to create environment for both data processing and training/inferencing. 
It may take quite a long time. 
(Change to your working directory when running commands if necessary, 
e.g., conda env create -f /home/yunya/environment.yml)

Link for creating environment by .yml (detailed): https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment
Link (short): https://shandou.medium.com/export-and-create-conda-environment-with-yml-5de619fe5a2

3. If it does not work, try to install environment based on the following steps.
Then, follow the instructions in Data_Processing_SAM.ipynb (Remember to change working directory at the beginning) to start processing data and SAM_Adapter_For_Building_Extraction.ipynb to start training/inferencing. 

4. Attention: if you download the packages directly from Colab, it is possible that you will miss the “pretrained” folder as shown below. If it happens, please download this folder manually.

5.Steps of installing codes for processing data for SAM
CV2 requires Python (3.7<=Python<3.11)
# install Python (or other names if necessary)
conda create -n sam  python==3.10
conda activate sam 

# install Jupyter lab related libraries
conda install -c conda-forge jupyterlab -y
conda install -c conda-forge nb_conda_kernels -y
conda install ipywidgets -y
conda install -c anaconda ipykernel -y
python -m ipykernel install --user --name data --display-name "data_sam (3.10)"

# install libraries related to data processing
conda install tqdm -y
conda install rasterio -y
conda install scipy -y
conda install imagecodecs -y
conda install scikit-learn -y
conda install scikit-image -y
conda install -c conda-forge opencv -y
conda install -c conda-forge gdal -y
conda install -c conda-forge proj geopandas -y
conda install -c conda-forge geopandas -y

6.Steps of installing codes for training SAM and inferencing
conda install pytorch==2.0.1 -y
conda install tensorboardX -y
conda install -c conda-forge segment-anything -y
pip install /home/yunya/anaconda3/envs/mmcv-1.7.0.tar.gz (change path if necessary)
conda install leafmap -y

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
