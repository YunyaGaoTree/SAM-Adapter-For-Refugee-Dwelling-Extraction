import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import glob
from skimage import io
from pprint import pprint
import re
import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def read_pred_gt_data(path_pred, path_gt): 
    
    # read data from path
    pred = io.imread(path_pred)
    gt = io.imread(path_gt)
    
    # process pred and gt
    new_width = min(pred.shape[0], gt.shape[0]) # perhaps one pixel shift, so to make their shape the same
    new_height = min(pred.shape[1], gt.shape[1])

    pred = pred[:new_width, :new_height]
    gt = gt[:new_width, :new_height]

    # turn 255 in gt or pred to 1
    pred[pred>0] = 1
    gt[gt>0] = 1
    
    return pred, gt


def visualize_diff(pred, gt, thres, path_out):
    
    # diff = gt - pred
    colour_dict = {-1: colors.to_rgb('crimson'), # false positive
                    0: colors.to_rgb('gainsboro'),
                    1: colors.to_rgb('blue')} # false negative
    
    colours_rgb = [colour_dict[i] for i in [-1, 0, 1]]
    colours_rgb = colors.ListedColormap(colours_rgb)

    # here, the values of gt and pred should be 0 and 1.
    # the values of diff: -1 (false positive), 0, 1 (false negative)
    diff = gt - pred 
    
    plt.figure(figsize=(50, 50))
    plt.axis('off')
    
    path_out_diff = os.path.join(path_out, "diff.png")
    plt.imshow(diff, cmap=colours_rgb, vmin=-1, vmax=1)
    plt.savefig(path_out_diff, bbox_inches='tight')
    
    path_out_gt = os.path.join(path_out, "gt.png")
    plt.imshow(gt)
    plt.savefig(path_out_gt, bbox_inches='tight')
    
    path_out_pred = os.path.join(path_out, "mask_pred_binary"+str(thres)+".png")
    plt.imshow(pred)
    plt.savefig(path_out_pred, bbox_inches='tight')


def calculate_iou(pred, gt):

    # IoU calculation = intersection / uniion
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union)
    
    return iou_score


def calculate_precision(pred, gt):
    
    # precision = TP (true predicted postive) / AP (all positive in gt)
    TP = np.sum(np.logical_and(pred, gt))
    AP = np.sum(pred)
    precision_score = TP / (AP + 1e-6)
    
    return precision_score


def calculate_recall(pred, gt):
    
    # recall = TP / PP (predicted postive)
    TP = np.sum(np.logical_and(pred, gt))
    PP = np.sum(gt) 
    recall_score = TP / (PP + 1e-6)
    
    return recall_score


def calculate_f1(pred, gt):
    
    # f1 = 2 * precision * recall / (precision + recall)
    precision = calculate_precision(pred, gt)
    recall = calculate_recall(pred, gt)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    
    return f1_score


def evaluate_metrics(pred, gt):
    
    eval_dict = {}
    
    recall = calculate_recall(pred, gt)
    precision = calculate_precision(pred, gt)
    f1= calculate_f1(pred, gt)
    iou = calculate_iou(pred, gt)
    
    eval_dict["recall"] = recall
    eval_dict["precision"] = precision
    eval_dict["f1"] = f1
    eval_dict["iou"] = iou
    
    return eval_dict
 
    
def filter_epoch_folder(path_list):
    
    def sort_key(s):
        # Extract the numerical part if the string starts with 'epoch'
        if s.startswith('epoch'):
            return int(re.findall(r'\d+', s)[0])
        # Return a high value for other strings so that they come at the end
        return float('inf')

    sorted_files = sorted(path_list, key=sort_key)

    sorted_files_epoch = []

    for item in sorted_files:
        if item.startswith("epoch"):
            sorted_files_epoch.append(item)    

    return sorted_files_epoch
    

# main function for predicted results from many epochs
# path_pred is the parent directory of all results from different epochs
# to create a csv to record all of the results 
# def evaluation_all_epochs_main(path_pred, path_gt, thres):
    
#     epoch_list = os.listdir(path_pred)
#     sorted_epoch_list = filter_epoch_folder(epoch_list)
    
#     epoch_namelist = [ep_name[5:] for ep_name in sorted_epoch_list]
    
#     accuracy_results = {
#                         'epoch': epoch_namelist,
#                         'f1': [],
#                         'iou': [],
#                         'recall': [],
#                         'precision': []
#                         }


#     for epoch in sorted_epoch_list:

#         # path of predicted results
#         path_pred_out = os.path.join(path_pred, epoch, "pred_mask_bin"+str(thres)+".tif")
#         path_pred_diff = os.path.join(path_pred, epoch)

#         # read predicted binary mask and gt mask
#         pred, gt = read_pred_gt_data(path_pred_out, path_gt)

#         # evaluate 
#         eval_result = evaluate_metrics(pred, gt)

#         accuracy_results["recall"].append(eval_result["recall"])
#         accuracy_results["precision"].append(eval_result["precision"])
#         accuracy_results["f1"].append(eval_result["f1"])
#         accuracy_results["iou"].append(eval_result["iou"])

#         # visualize images, gt, and predicted binary mask
#         visualize_diff(pred, gt, thres, path_pred_diff)

#     # Create the DataFrame
#     df = pd.DataFrame.from_dict(accuracy_results)

#     # Save to CSV
#     csv_file = os.path.join(path_pred, 'accuracy_results.csv')
#     df.to_csv(csv_file, index=False)

#     print(f'{csv_file} has been created successfully!')

def evaluation_single_main(path_pred, path_gt, thres=0.5):
    
    # read predicted binary mask and gt mask
    pred, gt = read_pred_gt_data(path_pred, path_gt)

    # evaluate 
    eval_result = evaluate_metrics(pred, gt)

    for key, value in eval_result.items():
        print(f'{key}: {value}')
        
    # elif size == "noFT":
    #     # path of predicted results
    #     path_pred_out = os.path.join(path_pred, "pred_mask_noFT.tif")
        
    #     # read predicted binary mask and gt mask
    #     pred, gt = read_pred_gt_data(path_pred_out, path_gt)

    #     # evaluate 
    #     eval_result = evaluate_metrics(pred, gt)
        
    #     # visualize images, gt, and predicted binary mask
    #     visualize_diff(pred, gt, thres, path_pred)
            
    #     print(path_pred_out)
    #     pprint(eval_result)
        
    
# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data', default=None, help="Dagaha2017 Djibo2019...")
#     parser.add_argument('--upsample', default="1024", help="1024 or SR") 
#     parser.add_argument('--size', default="small", help="small or large or noFT") 
#     parser.add_argument('--uptype', default="", help="nearest bilinear EDSR") 
#     parser.add_argument('--thres', default=0.5, help="the threshold to determine the binary map") 
    
#     # path_data = "path of your data folder" # e.g. "/home/usr/Data"
#     path_data = "/home/yunya/anaconda3/envs/Data"
    
#     # read prompts
#     args = parser.parse_args()
#     data_name = args.data
#     size = args.size
#     upsample = args.upsample
#     uptype = args.uptype
#     thres = args.thres

#     # testing data
#     if upsample == "1024":
#         path_test_data = os.path.join(path_data, data_name, "raw", "test")
#         path_pred = os.path.join('outputs', data_name, size, upsample)
        
#     elif upsample == "SR":
#         path_test_data = os.path.join(path_data, data_name, "SAM", upsample, "test")
#         path_pred = os.path.join('outputs', data_name, size, upsample, uptype)
    
#     pathlib.Path(path_pred).mkdir(parents=True, exist_ok=True)
#     path_gt = glob.glob(os.path.join(path_test_data, "gt") + "/*.tif")[0]

#     # run the main function
#     main(path_pred, path_gt, thres)
    