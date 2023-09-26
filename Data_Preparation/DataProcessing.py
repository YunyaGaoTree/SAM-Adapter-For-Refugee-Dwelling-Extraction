# Transform Inria gt dataset (binary image) to COCO format
# Using cv2.findcontours and polygon simplify to convert raster label to vector label

from pycocotools.coco import COCO
import os
import numpy as np
from skimage import io
import json
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Polygon, mapping
from skimage.measure import label as ski_label
from skimage.measure import regionprops
from shapely.geometry import box
import cv2
import glob
import math
import subprocess
import sys
import scipy
import pathlib
import rasterio
from rasterio.enums import Resampling
from PIL import Image


def polygon2hbb(poly):
    """
    Get horizontal bounding box (match COCO)
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    hbb_x = np.min(p_x)
    hbb_y = np.min(p_y)
    hbb_w = np.around(np.max(p_x) - hbb_x, decimals=2)
    hbb_h = np.around(np.max(p_y) - hbb_y, decimals=2)
    hbox = [hbb_x, hbb_y, hbb_w, hbb_h]
    return [float(i) for i in hbox]

def clip_by_bound(poly, im_h, im_w):
    """
    Bound poly coordinates by image shape
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    p_x = np.clip(p_x, 0.0, im_w-1)
    p_y = np.clip(p_y, 0.0, im_h-1)
    return np.concatenate((p_x[:, np.newaxis], p_y[:, np.newaxis]), axis=1)

def crop2patch(im_p, p_h, p_w, p_overlap):
    """
    Get coordinates of upper-left point for image patch
    return: patch_list [X_upper-left, Y_upper-left, patch_width, patch_height]
    """
    im_h, im_w, _ = im_p
    x = np.arange(0, im_w-p_w, p_w-p_overlap)
    x = np.append(x, im_w-p_w)
    y = np.arange(0, im_h-p_h, p_h-p_overlap)
    y = np.append(y, im_h-p_h)
    X, Y = np.meshgrid(x, y)
    patch_list = [[i, j, p_w, p_h] for i, j in zip(X.flatten(), Y.flatten())]
    return patch_list

def polygon_in_bounding_box(polygon, bounding_box):
    """
    Returns True if all vertices of polygons are inside bounding_box
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    result = np.all(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[0] + bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[1] + bounding_box[3])
        )
    )
    return result

def transform_poly_to_bounding_box(polygon, bounding_box):
    """
    Transform the original coordinates of polygon to bbox
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    transformed_polygon = polygon.copy()
    transformed_polygon[:, 0] -= bounding_box[0]
    transformed_polygon[:, 1] -= bounding_box[1]
    return transformed_polygon

def bmask_to_poly(b_im, simplify_ind, tolerance=1.8, ):
    """
    Convert binary mask to polygons
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    try:
        label_img = ski_label(b_im > 0)
    except:
        print('error')
    props = regionprops(label_img)
    for prop in props:
        prop_mask = np.zeros_like(b_im)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(prop_mask, pad_width=1, mode='constant', constant_values=0)
        contours, hierarchy = cv2.findContours(padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            intp = []
            for contour, h in zip(contours, hierarchy[0]):
                contour = np.array([c.reshape(-1).tolist() for c in contour])
                # subtract pad
                contour -= 1
                contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
                if len(contour) > 3:
                    closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                    if h[3] < 0:
                        extp = [tuple(i) for i in closed_c]
                    else:
                        if cv2.contourArea(closed_c.astype(int)) > 10:
                            intp.append([tuple(i) for i in closed_c])
            poly = Polygon(extp, intp)
            if simplify_ind:
                poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
        elif len(contours) == 1:
            contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
            contour -= 1
            contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
            if len(contour) > 3:
                closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                poly = Polygon(closed_c)

            # simply polygon vertex
                if simplify_ind:
                    poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
            # print(np.array(poly.exterior.coords).ravel().tolist())
            # in case that after "simplify", one polygon turn to multiply polygons
            # (pixels in polygon) are not connected
    return polygons

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_crop(im, gt, crop_size, angle):
    h, w = im.shape[0:2]
    im_rotated = rotate_image(im, angle)
    gt_rotated = rotate_image(gt, angle)
    if largest_rotated_rect(w, h, math.radians(angle))[0] > crop_size:
        im_cropped = crop_around_center(im_rotated, crop_size, crop_size)
        gt_cropped = crop_around_center(gt_rotated, crop_size, crop_size)
    else:
        print('error')
        im_cropped = crop_around_center(im, crop_size, crop_size)
        gt_cropped = crop_around_center(gt, crop_size, crop_size)
    return im_cropped, gt_cropped


def lt_crop(im, gt, crop_size):
    im_cropped = im[0:crop_size, 0:crop_size, :]
    gt_cropped = gt[0:crop_size, 0:crop_size]
    return im_cropped, gt_cropped


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
    
    
def contrast98_2_convert2uint8(image_data):
    
    minval = np.percentile(image_data, 2)
    maxval = np.percentile(image_data, 98)
    pixvals = np.clip(image_data, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 255
    pixvals = pixvals.astype(np.uint8)
    
    return pixvals
    
def data_process_hisup(path_dataset, data_type, patch_size=512, model_type="HiSup"):
    # data_type: train_large, train_small, test, val, etc
    # install("imagecodecs")

    input_image_path = path_dataset + '/raw/'+data_type+'/images/'
    input_gt_path = path_dataset + '/raw/'+data_type+'/gt/'
    save_path = path_dataset + '/'+model_type+'/'+data_type+'/' 

    val_set = [] # we seperate the validation data, so it is not necessary

    output_im_train = os.path.join(save_path, 'images')
    if not os.path.exists(output_im_train):
        os.makedirs(output_im_train)

    patch_width = int(patch_size * 1.5)
    patch_height = int(patch_size * 1.5)
    patch_overlap = int(patch_size * 0.4)
    
    if "train" in data_type:
        rotation_list = [67.5] #add more if having more memory resources
    elif data_type == "val" or data_type == "test":
        rotation_list = []

    # main dict for annotation file
    output_data_train = {
        'info': {'description': 'building footprints', 'contributor': 'YG'},
        'categories': [{'id': 1, 'name': 'building'}],
        'images': [],
        'annotations': [],
    }

    train_ob_id = 0
    train_im_id = 0
    # read in data with npy format
    input_label = os.listdir(input_gt_path)
    for g_id, label in enumerate(tqdm(input_label)):
        # read data
        label_info = [''.join(list(g)) for k, g in groupby(label, key=lambda x: x.isdigit())]
        label_name = label_info[0] + label_info[1]
        image_data = io.imread(os.path.join(input_image_path, label_name + '.tif'))
        
        if image_data.shape[2] == 4:
            image_data = select_bands(image_data)
        
        image_data = contrast98_2_convert2uint8(image_data)
        
        gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.tif'))
        gt_im_data = gt_im_data.astype("uint8") 
        gt_im_data[0, :] = 0 # because some of images with a boundary with wrong values
        gt_im_data[:, 0] = 0
        gt_im_data[:, -1] = 0
        gt_im_data[-1, :] = 0

        im_h, im_w, _ = image_data.shape

        # for training set, split image to small patches, e.g. 512x512
        patch_list = crop2patch(image_data.shape, patch_width, patch_height, patch_overlap)
        for pid, pa in enumerate(patch_list):
            x_ul, y_ul, pw, ph = pa

            p_gt = gt_im_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width]
            p_im = image_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width, :]
            p_gts = []
            p_ims = []
            p_im_rd, p_gt_rd = lt_crop(p_im, p_gt, patch_size)
            p_gts.append(p_gt_rd)
            p_ims.append(p_im_rd)
            for angle in rotation_list:
                rot_im, rot_gt = rotate_crop(p_im, p_gt, patch_size, angle)
                p_gts.append(rot_gt)
                p_ims.append(rot_im)
            for p_im, p_gt in zip(p_ims, p_gts):
                p_gt = p_gt.astype('uint8')
                if np.sum(p_gt > 0) > 5:
                    p_polygons = bmask_to_poly(p_gt, 1)
                    for poly in p_polygons:
                        p_area = round(poly.area, 2)
                        if p_area > 0:
                            p_bbox = [poly.bounds[0], poly.bounds[1],
                                      poly.bounds[2]-poly.bounds[0], poly.bounds[3]-poly.bounds[1]]
                            if p_bbox[2] > 5 and p_bbox[3] > 5:
                                p_seg = []
                                coor_list = mapping(poly)['coordinates']
                                for part_poly in coor_list:
                                    p_seg.append(np.asarray(part_poly).ravel().tolist())
                                anno_info = {
                                    'id': train_ob_id,
                                    'image_id': train_im_id,
                                    'segmentation': p_seg,
                                    'area': p_area,
                                    'bbox': p_bbox,
                                    'category_id': 100,
                                    'iscrowd': 0
                                }
                                output_data_train['annotations'].append(anno_info)
                                train_ob_id += 1
                # get patch info
                p_name = label_name + '-' + str(train_im_id) + '.tif'
                patch_info = {'id': train_im_id, 'file_name': p_name, 'width': patch_size, 'height': patch_size}
                output_data_train['images'].append(patch_info)
                # save patch image
                io.imsave(os.path.join(output_im_train, p_name), p_im)
                train_im_id += 1

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'annotation.json'), 'w') as f_json:
        json.dump(output_data_train, f_json)
        

def data_process_sam_seg(path_dataset, data_type, patch_size, model_type):
    # data_type: train or test
    # install("imagecodecs")

    input_image_path = path_dataset+'/raw/'+data_type+'/images/'
    input_gt_path = path_dataset+'/raw/'+data_type+'/gt/'
    save_path = path_dataset+'/'+model_type+'/'+str(patch_size)+'/'+data_type+'/' 

    val_set = [] 

    output_im_train = os.path.join(save_path, 'images')
    output_gt_train = os.path.join(save_path, 'gt')
    if not os.path.exists(output_im_train):
        os.makedirs(output_im_train)
        os.makedirs(output_gt_train)

    # the selection of patch width or height or overlap is important, if width and height are too small, then some errors may occur.
    # this part can be modified based on different purposes (more data or less data to be created)
    if patch_size == 1024:
        patch_width = 1500
        patch_height = 1500
        patch_overlap = 400
    
    elif patch_size == 256:
        patch_width = 400
        patch_height = 400
        patch_overlap = 100
        
    else:
        patch_width = int(patch_size * 1.5)
        patch_height = int(patch_size * 1.5)
        patch_overlap = int(patch_size * 0.5)
        
    if "train" in data_type:
        rotation_list = [67.5]  #add more if having more memory resources
    elif data_type == "val" or data_type == "test":
        rotation_list = []

    # main dict for annotation file
    output_data_train = {
        'info': {'description': 'building footprints', 'contributor': 'YG'},
        'categories': [{'id': 1, 'name': 'building'}],
        'images': [],
        'annotations': [],
    }

    train_ob_id = 0
    train_im_id = 0
    # read in data with npy format
    input_label = os.listdir(input_gt_path)
    for g_id, label in enumerate(tqdm(input_label)):
        # read data
        label_info = [''.join(list(g)) for k, g in groupby(label, key=lambda x: x.isdigit())]
        label_name = label_info[0] + label_info[1]
        
        image_data = io.imread(os.path.join(input_image_path, label_name + '.tif'))
        image_data = contrast98_2_convert2uint8(image_data)
        
        gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.tif'))
        gt_im_data[0, :] = 0 # because some of images with a boundary with wrong values
        gt_im_data[:, 0] = 0
        gt_im_data[:, -1] = 0
        gt_im_data[-1, :] = 0
        gt_im_data = gt_im_data.astype("uint8")
        
        im_h, im_w, _ = image_data.shape

        patch_list = crop2patch(image_data.shape, patch_width, patch_height, patch_overlap)
        for pid, pa in enumerate(patch_list):
            x_ul, y_ul, pw, ph = pa

            p_gt = gt_im_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width]
            p_im = image_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width, :]
            p_gts = []
            p_ims = []
            p_im_rd, p_gt_rd = lt_crop(p_im, p_gt, patch_size)
            p_gts.append(p_gt_rd)
            p_ims.append(p_im_rd)
            
            for angle in rotation_list:
                rot_im, rot_gt = rotate_crop(p_im, p_gt, patch_size, angle)
                p_gts.append(rot_gt)
                p_ims.append(rot_im)
            for p_im, p_gt in zip(p_ims, p_gts):
                
                p_gt = p_gt.astype('uint8')
                p_gt[p_gt > 0] = 255
                
                # get patch info
                p_name = label_name + '-' + str(train_im_id) + '.png' # it is important to use png, otherwise the value of input array will be chanaged
                patch_info = {'id': train_im_id, 'file_name': p_name, 'width': patch_size, 'height': patch_size}
                output_data_train['images'].append(patch_info)
                
                # save patch image
                io.imsave(os.path.join(output_im_train, p_name), p_im, check_contrast=False)
                
                # save patch GT
                p_gt_ = np.expand_dims(p_gt, axis=-1)
                p_gt_3ch = np.concatenate((p_gt_, p_gt_, p_gt_), axis=-1)
                io.imsave(os.path.join(output_gt_train, p_name), p_gt_3ch, check_contrast=False)
                
                train_im_id += 1

#### Super resolution ####
                
import cv2
    
def set_sr_model():
    # if cannot import dnn_superres from cv2, then: 
    # pip install --upgrade opencv-python
    # pip install --upgrade opencv-contrib-python

    # links to download models
    # https://github.com/Saafke/EDSR_Tensorflow/blob/master/models/
    # https://github.com/fannymonori/TF-LapSRN/blob/master/export/LapSRN_x8.pb
    # feel free to use other SR models, based on the experiences here, EDSR is the most efficient pretrained model compared to other models

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

def upscale_img(path_in, path_out, sr_model):
    
    # load the image
    image = cv2.imread(path_in)

    # upsample the image
    upscaled = sr_model.upsample(image)

    # save the upscaled image
    cv2.imwrite(path_out, upscaled)

    
def upscale_lab(path_in, path_out):
    
    # load the image
    lab = cv2.imread(path_in)

    # upsample the image
    upscale_times = 4 # ratio of original size
    width = int(lab.shape[1] * upscale_times)
    height = int(lab.shape[0] * upscale_times)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(lab, dim)
    resized[resized>0] = 255

    # save the upscaled image
    cv2.imwrite(path_out, resized)
    
def upscale_testing_data_SR(path_database, dataset, upscaled_folder, data_type):
    
    dtype = "test"
    model_name = "SAM"
    upscale_factor = 4

    path_test_img = os.path.join(path_database, dataset, "raw", "test", data_type)
    path_in_img = glob.glob(path_test_img + "/*.tif")[0]

    path_out= os.path.join(path_database, dataset, model_name, "SR", dtype, upscaled_folder)
    path_out_img = os.path.join(path_out, "test_up.tif")
    pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
    print(path_out_img)
    
    sr_model = set_sr_model(upscaled_folder)
    
    upscaled_img = upscale_image_by_SR(path_in_img, sr_model)
    
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
                    

#### Flip + Rotation to augment small dataset ####

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
    print(operation + ": images saved.")
    
    augmentation_single(gt_list, dtype, operation, degrees, op_times, path_base_gt)
    print(operation + ": gt saved.")