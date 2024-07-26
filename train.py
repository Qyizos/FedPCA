# -*- coding: utf-8 -*-
import sys, os
import logging
import glob
import time
from torch.utils import data
from torch.utils.data import dataset

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from federated.configs import set_configs

from federated.fl_api import *
from federated.model_trainer_segmentation import ModelTrainerSegmentation

from utils.dataset import DataFolder
from utils.my_transforms import get_transforms
from utils.readVal import getVal
from torch.utils.data import DataLoader
from nets.model import *
import test

def deterministic(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def set_paths(args):
    args.save_path = '/data/user/FedPA/experiments/{}'.format(args.sonName)
    exp_folder = '{}'.format(args.mode)

    if args.balance:
        exp_folder = exp_folder + '_balanced'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

def custom_model_trainer(args, model=ResUNet34(pretrained = True)):
    model_trainer = ModelTrainerSegmentation(model, args)

    return model_trainer

def custom_federated_api(args, model_trainer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    federated_api = FedPAAPI(device, args, model_trainer)
    return federated_api

def main(curTime=None, curmodel=ResUNet34(pretrained = True), notes=None):
    args = set_configs()
    args.generalize = False
    args.source = ['MO']
    args.notes = notes
    args.transform = dict()
    args.transform['train'] = {
        'random_resize': [0.8, 1.25],
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_affine': 0.3,
        'random_rotation': 90,
        'random_crop': args.input_size,
        'label_encoding': 2,
        'to_tensor': 3
    }
    args.transform['test'] = {
        'to_tensor': 3
    }

    valDSet = args.source[0].split('_')[0]
    valDSetList = [valDSet]

    deterministic(args.seed)
    if curTime:
        args.sonName = curTime
    else:
        args.sonName = time.strftime('%y%m%d-%H%M', time.localtime())
    set_paths(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_trainer = custom_model_trainer(args, curmodel)

    model = curmodel
    model = model.cuda()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    args.num_params = num_params

    cudnn.benchmark = True

    log_path = os.path.join(args.save_path, 'log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.9, 0.99),
                                 weight_decay=1e-4)
    criterion = torch.nn.NLLLoss(ignore_index=2).cuda()

    # ----- load data ----- #
    data_transforms = {'train': get_transforms(args.transform['train']),
                       'test': get_transforms(args.transform['test'])}

    dir_list = ['images', 'imgDepth_dir', 'imgMix_dir', 'labels_cluster', 'labels_voronoi']
    post_fix = ['.png', '.png', '_label_vor.png', '_label_cluster.png']
    num_channels = [3, 1, 3, 3, 3]
    datasets = []
    for client in args.source:
        train_set = DataFolder(dir_list, post_fix, num_channels, client, data_transforms['train'])
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=1)
        datasets.append(train_loader)

    valSets = []
    for curClient in args.source:
        valSet = getVal(client = curClient, type='val', enhancetype = 'GA0')
        valSets.append(valSet)
    federated_manager = custom_federated_api(args, model_trainer)
    federated_manager.train(datasets, model, optimizer, criterion, args, valSets)

    log_dir = args.save_path

    log_files = glob.glob(os.path.join(log_dir, 'best_val_*.pth.tar'))
    log_files.sort(key=os.path.getmtime)
    files_to_remove = len(log_files) - 1

    for i in range(files_to_remove):
        os.remove(log_files[i])
        print(f"Deleted: {log_files[i]}")

    log_files = glob.glob(os.path.join(log_dir, 'best_val_*.pth.tar'))
    log_files.sort(key=os.path.getmtime)
    for i in log_files:
        bestName = i.split('/')[-1].split('.')[0]
        test.main(curTime, curmodel, path=bestName, testSet=valDSetList)

if __name__ == "__main__":
    curTime = time.strftime('%y%m%d-%H%M', time.localtime())

    notes = 'model = ResUNet34_concat5L_proFsV4k_OGC3_OSb2_fs5(); '
    name2 = notes.split('model = ')[-1].split('()')[0]
    curTime = curTime + '_' + name2

    main(curTime, ResUNet34_concat5L_proFsV4k_OGC3_OSb2_fs5(), notes)
