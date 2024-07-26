# -*- coding: utf-8 -*-
import sys, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure
from scipy import misc
import imageio
import logging
from nets.model import *
from nets.SegNet_cons import SegNet
import utils.utils as utils
from utils.accuracy import compute_metrics
import time

from utils.my_transforms import get_transforms
from federated.configs import set_configs

def main(curTime=None, curmodel=ResUNet34_concat5L_proFsV4k_OGC3_OSb2_fs5(), path='best_val_239',testSet=['TNBC']):
    args = set_configs()
    min_area = 20
    patch_size = 224
    overlap = 80
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    datsSource = testSet

    if path == 'final':
        pathName = 'final.pth.tar'
    elif path == 'best_loss':
        pathName = 'best_loss.pth.tar'
    elif path == 'best_val':
        pathName = 'best_val.pth.tar'
    else:
        pathName = path + '.pth.tar'

    if curTime:
        model_path = '/data/user/FedPA/experiments/' + curTime + '/' + pathName
    else:
        model_path = '/data/user/FedPA/experiments/projectName/best.pth.tar'
    parent_path = os.path.dirname(model_path)  
    log_path = os.path.join(parent_path, 'test_log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    args.transform = dict()
    args.transform['train'] = {
        'random_resize': [0.8, 1.25],
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_affine': 0.3,
        'random_rotation': 90,
        'random_crop': args.input_size,
        'label_encoding': 2,
        'to_tensor': 1
    }
    args.transform['test'] = {
        'to_tensor': 1
    }

    # data transforms
    test_transform = get_transforms(args.transform['test'])

    
    model = curmodel
    
    model = model.cuda()
    cudnn.benchmark = True 

    # ----- load trained model ----- #
    print("=> loading trained model")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    

    # switch to evaluate mode
    model.eval() 
    counter = 0
    print("=> Test begins:")

    for curData in datsSource:
        img_dir = '/data/user/FedPA/data_for_train/{:s}/images/test'.format(curData)
        img_MixDir = '/data/user/FedPA/data_for_train/{:s}/images/test_Mix'.format(curData)

        img_depthDir = img_MixDir
        testPath = curData.split('-')[0]
        label_dir = './data/{:s}/labels_instance'.format(testPath)

        save_dir = os.path.join(parent_path, 'test_results')

        logging.info("############ curData：{:s}, {:s}#############".format(curData, path))

        img_names = os.listdir(img_dir)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        curSaveDir = '{:s}/{:s}'.format(save_dir, curData)
        if not os.path.exists(curSaveDir):
            os.mkdir(curSaveDir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}/{:s}_prob_maps'.format(save_dir, curData, strs[-1])
        seg_folder = '{:s}/{:s}/{:s}_segmentation'.format(save_dir, curData, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

        metric_names = ['acc', 'p_F1', 'dice', 'aji']
        test_results = dict()
        all_result = utils.AverageMeter(len(metric_names))

        for img_name in img_names:
            # load test image
            print('=> Processing image {:s}'.format(img_name))
            img_path = '{:s}/{:s}'.format(img_dir, img_name)
            img = Image.open(img_path)

            imgDepth_path = '{:s}/{:s}'.format(img_depthDir, img_name)
            imgMix_path = '{:s}/{:s}'.format(img_MixDir, img_name)
            imgDepth = Image.open(imgDepth_path)
            imgMix = Image.open(imgMix_path)

            ori_h = img.size[1]
            ori_w = img.size[0]
            name = os.path.splitext(img_name)[0]
            label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
            gt = imageio.imread(label_path)

            input = test_transform((img,))[0].unsqueeze(0)
            inputMix = test_transform((imgMix,))[0].unsqueeze(0)

            print('\tComputing output probability maps...')
            prob_maps = get_probmaps2(input, inputMix, model, patch_size, overlap)
            pred = np.argmax(prob_maps, axis=0)

            pred_labeled = measure.label(pred)
            pred_labeled = morph.remove_small_objects(pred_labeled, min_area)
            pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
            pred_labeled = measure.label(pred_labeled)

            print('\tComputing metrics...')
            metrics = compute_metrics(pred_labeled, gt, metric_names)

            # save result for each image
            test_results[name] = [metrics['acc'], metrics['p_F1'], metrics['dice'], metrics['aji']]

            # update the average result
            all_result.update([metrics['acc'], metrics['p_F1'], metrics['dice'], metrics['aji']])

            # save image
            print('\tSaving image results...')
            misc.imsave('{:s}/{:s}_pred.png'.format(prob_maps_folder, name), pred.astype(np.uint8) * 255)
            misc.imsave('{:s}/{:s}_prob.png'.format(prob_maps_folder, name), prob_maps[1, :, :])

            final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
            final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

            # save colored objects
            pred_colored_instance = np.zeros((ori_h, ori_w, 3))
            for k in range(1, pred_labeled.max() + 1):
                pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            misc.imsave(filename, pred_colored_instance)

            counter += 1
            if counter % 10 == 0:
                print('\tProcessed {:d} images'.format(counter))

        logging.info('=> Processed all {:d} images'.format(counter))
        logging.info(
            'Average Acc: {r[0]:.4f}\nF1: {r[1]:.4f}\nDice: {r[3]:.4f}\nAJI: {r[4]:.4f}\n'.format(
                r=all_result.avg))

    header = metric_names
    utils.save_results(header, all_result.avg, test_results, '{:s}/test_results.txt'.format(save_dir))


def get_probmaps2(input, input2, model, patch_size, overlap):
    size = patch_size
    overlap = overlap

    if size == 0:
        with torch.no_grad():
            output = model(input.cuda(), input2.cuda())
    else:
        output = utils.split_forward2(model, input, input2, size, overlap)
    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


if __name__ == '__main__':
    main()
