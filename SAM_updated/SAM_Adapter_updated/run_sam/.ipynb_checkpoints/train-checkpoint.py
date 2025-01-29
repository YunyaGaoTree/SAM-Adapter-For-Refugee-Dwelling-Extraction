import sys
import os

# Get the parent directory of the current script
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the Python path
sys.path.append(parent_directory)

import warnings
"""the version of mmcv used here is 1.7. current updated version is 2.0"""
warnings.filterwarnings("ignore", category=DeprecationWarning, module='mmcv')

import argparse
import yaml
import pathlib
import glob
import shutil

import datasets
import models
import utils
from statistics import mean

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.distributed as dist

torch.distributed.init_process_group(backend='nccl') 

from inference_ft import inference_main

def write_yaml(data_yaml, path_out):
    """ A function to write YAML file"""
    
    path_out_yaml = os.path.join(path_out, "config.yaml")
    with open(path_out_yaml, 'w') as f:
        yaml.dump(data_yaml, f)
        

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True, sampler=None)
    return loader


def make_data_loaders(config):
    
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    # val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader #, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    pbar = tqdm(total=len(loader), leave=False, desc='val')

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4


def prepare_training(config):
    
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
        
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model):
    
    model.train()

    pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    
    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']

        inp_npy = inp.cpu().numpy()
        gt_npy = gt.cpu().numpy()
        print(inp_npy)
        print(gt_npy)
        
        model.set_input(inp, gt)
        model.optimize_parameters()
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def save(config, model, save_path, name):
    
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{name}.pth"))


def delete_checkpoint_folder_datafolder(path_data):
    
    img_folder_path = os.path.join(path_data, "images", ".ipynb_checkpoints")
    gt_folder_path = os.path.join(path_data, "gt", ".ipynb_checkpoints")

    if os.path.exists(img_folder_path):
        shutil.rmtree(img_folder_path)
        print("ipynb_checkpoints folder of images deleted successfully.")
    elif os.path.exists(gt_folder_path):
        shutil.rmtree(gt_folder_path)
        print("ipynb_checkpoints folder deleted successfully.")
    else:
        print("ipynb_checkpoints folder not found.")
    

def main(config, path_output, path_model, model_save_epoch, inference_save_epoch, thres, upsample, path_test_img_list, path_test_gt_list):
    
    global log, writer, log_info
    log, writer = utils.set_save_path(path_output, remove=False)
    
    # train_loader, val_loader = make_data_loaders(config)
    train_loader = make_data_loaders(config)

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training(config)
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    sam_checkpoint = torch.load(config['sam_checkpoint'], weights_only=True)
    model.load_state_dict(sam_checkpoint, strict=False)
    
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    model.train()
    model.cuda()
    
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    
    for epoch in range(epoch_start, epoch_max + 1):
        
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model)
        lr_scheduler.step()

        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        log_info.append('train G: loss={:.4f}'.format(train_loss_G))
        writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

        model_spec = config['model']
        model_spec['sd'] = model.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        
        # save the trained model
        if epoch % model_save_epoch == 0:
            save(config, model, path_model, 'epoch'+str(epoch))
        
        model.eval()

        # inference - testing data
        if epoch % inference_save_epoch == 0:
      
            path_output_pred = os.path.join(path_output, "epoch"+str(epoch))
            pathlib.Path(path_output_pred).mkdir(parents=True, exist_ok=True)
            
            with torch.no_grad():
                inference_main(model, config, thres, path_output_pred, upsample, path_test_img_list, path_test_gt_list)
        
        model.train()
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/config_sam_vit_h.yaml", help="use the hyperparameters provided by SAM-Adapter")
    parser.add_argument('--data', default=None, help="different datasets")
    parser.add_argument('--upsample', default="1024", help="1024 or upscaled") 
    parser.add_argument('--size', default="small", help="small or large") 
    parser.add_argument('--uptype', default="", help="cubic or SR") 
    parser.add_argument('--epoch', default=10, help="epochs for training") 
    parser.add_argument('--model_save_epoch', default=2, help="the interval of saving trained models, do not save models in default due to big size of model.") 
    parser.add_argument('--inference_save_epoch', default=1, help="the interval of saving trained models") 
    parser.add_argument('--thres', default=0.5, help="the threshold to determine the binary map") 
    
    # path_data = "path of your data folder" # e.g. "/home/usr/Data"
    path_data = "/home/ubuntu/SAM1/Data" 
    
    args = parser.parse_args()
    path_cfg = args.config
    data_name = args.data
    size = args.size
    upsample = args.upsample
    uptype = args.uptype
    model_save_epoch = int(args.model_save_epoch)
    inference_save_epoch = int(args.inference_save_epoch)
    thres = args.thres
    epoch = int(args.epoch)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # open the base config file to read hyperparameters
    with open(path_cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # number of epochs for training
    config["epoch_max"] = epoch
    
    # training data
    path_train_data = os.path.join(path_data, data_name, "SAM", upsample)
    delete_checkpoint_folder_datafolder(path_train_data)
        
    config["train_dataset"]["dataset"]["args"]["root_path_1"] = os.path.join(path_train_data, uptype, "train_"+size, "images")
    config["train_dataset"]["dataset"]["args"]["root_path_2"] = os.path.join(path_train_data, uptype, "train_"+size, "gt")

    # testing data
    if upsample == "1024":
        path_test_data = os.path.join(path_data, data_name, "raw", "test")
        path_test_img_list = glob.glob(os.path.join(path_test_data, "images") + "/*.tif")
        path_test_gt_list = glob.glob(os.path.join(path_test_data, "gt") + "/*.tif")
        
    elif upsample == "upscaled":
        path_test_data = os.path.join(path_data, data_name, "SAM", upsample, "test")
        path_test_img_list = glob.glob(os.path.join(path_test_data, uptype, "images") + "/*.tif")
        path_test_gt_list = glob.glob(os.path.join(path_test_data, uptype, "gt") + "/*.tif")

    path_test_img_list.sort()
    path_test_gt_list.sort()
    
    # define path of outputs of predicted results and models to be saved
    path_output = os.path.join('outputs', data_name, size, upsample, uptype)
    path_model = os.path.join('save_model', data_name, size, upsample, uptype)

    pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_model).mkdir(parents=True, exist_ok=True)
    
    write_yaml(config, path_output)
    print('config saved.')
    
    main(config, path_output, path_model, model_save_epoch, inference_save_epoch, thres, upsample, path_test_img_list, path_test_gt_list)
