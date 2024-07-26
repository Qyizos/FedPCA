# -*- coding: utf-8 -*-
import copy,os,glob
import logging
import random
import sys,os
import numpy as np
import pandas as pd
import torch
import math
import utils.utils as utils
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from utils.accuracy import compute_metrics
from nets.model import *
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure
from typing import Dict, List, OrderedDict, Tuple
from scipy import misc
from torch.optim.lr_scheduler import StepLR

class FedPAAPI(object):
    def __init__(self, device, args, model_trainer):
        """
        dataset: data loaders and data size info
        """
        self.device = device
        self.args = args

        client_num = len(args.source)
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)

        self.client_list = args.source
        self.model_trainer = model_trainer

    def train(self, datasets, model, optimizer, criterion, args, valSet):
        w_globalPre = self.model_trainer.get_model_params()
        w_global = self.model_trainer.get_model_params()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_metrics = 0
        tb_writer = SummaryWriter('{:s}/tb_logs'.format(args.save_path))

        for round_idx in range(self.args.comm_round):

            logging.info("============ Communication round : {}".format(round_idx))
            w_locals = []
            sum_metricsList = [0, 0, 0]
            w_locals_Pre = []
            wEMA = 0.7
            curLoss = 0
            lossVar = 0
            lossCluster = 0

            metric_names = ['aji']
            val_result = utils.AverageMeter(len(metric_names))
            val_results = dict()

            for idx, CurData in enumerate(datasets):
                model.load_state_dict(copy.deepcopy(w_global))
                model.to(device)
                model.train()

                results = utils.AverageMeter(4)

                for i, sample in enumerate(CurData):
                    input, imgDepth, imgMix, target1, target2 = sample

                    if target1.dim() == 4:
                        target1 = target1.squeeze(1)
                    if target2.dim() == 4:
                        target2 = target2.squeeze(1)

                    target1 = target1.cuda()
                    target2 = target2.cuda()

                    input_var = input.cuda()
                    imgMix_var = imgMix.cuda()

                    output = model(input_var, imgMix_var)

                    log_prob_maps = F.log_softmax(output, dim=1)
                    loss_vor = criterion(log_prob_maps, target1)
                    loss_cluster = criterion(log_prob_maps, target2)

                    loss = loss_vor + loss_cluster

                    result = [loss.item(), loss_vor.item(), loss_cluster.item(), -1]

                    results.update(result, input.size(0))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % args.log_interval == 0:
                        logging.info('\tIteration: [{:d}/{:d}]'
                                    '\tLoss {r[0]:.4f}'
                                    '\tLoss_vor {r[1]:.4f}'
                                    '\tLoss_cluster {r[2]:.4f}'
                                    '\tLoss_CRF {r[3]:.4f}'.format(i, len(CurData), r=results.avg))
                logging.info('\t---------------------------------' + args.source[idx] + '---------------------------------')
                logging.info('\t=> Train Avg: Loss {r[0]:.4f}'
                            '\tloss_vor {r[1]:.4f}'
                            '\tloss_cluster {r[2]:.4f}'
                            '\tloss_CRF {r[3]:.4f}'.format(r=results.avg))

                if (round_idx + 1) > 1:
                    model = model.cuda()
                    model.eval()
                    min_area = 20
                    patch_size = 224
                    overlap = 80
                    for valIdx, CurVal in enumerate(valSet[idx]):
                        input, inputMix, gt = CurVal

                        prob_maps = self.get_probmaps2(input, inputMix, model, patch_size, overlap)
                        pred = np.argmax(prob_maps, axis=0)

                        pred_labeled = measure.label(pred)
                        pred_labeled = morph.remove_small_objects(pred_labeled, min_area)
                        pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
                        pred_labeled = measure.label(pred_labeled)

                        metrics = compute_metrics(pred_labeled, gt, metric_names)
                        sum_metricsList[idx] = sum_metricsList[idx] + metrics['aji']

                        val_results[valIdx] = [metrics['aji']]
                        val_result.update([metrics['aji']])

                    sum_metricsList[idx] = sum_metricsList[idx] / len(valSet[idx])
                    print(sum_metricsList[idx])

                curLoss = curLoss + results.avg[0]
                lossVar = curLoss + results.avg[1]
                lossCluster = curLoss + results.avg[2]

                w = model.cpu().state_dict()
                w_locals.append((len(CurData), copy.deepcopy(w), copy.deepcopy(model)))

            curLoss = curLoss / len(datasets)
            lossVar = lossVar / len(datasets)
            lossCluster = lossCluster / len(datasets)

            if (round_idx + 1) > 1:
                w_globalC = self._aggregatePath(w_locals, valSet, round_idx, sum_metricsList)

                w_locals_Pre.append(((1 - wEMA), copy.deepcopy(w_globalPre)))
                w_locals_Pre.append((wEMA, copy.deepcopy(w_globalC)))

                w_global = self._aggregateEMA(w_locals_Pre)

                model.load_state_dict(copy.deepcopy(w_global))

                saveName = 'curModel.pth.tar'
                save_mode_path = os.path.join(self.args.save_path, saveName)
                state = {
                    'epoch': round_idx + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                torch.save(state, save_mode_path)
            else:
                sum_metricsList = [1, 1, 1]
                w_global = self._aggregatePath(w_locals, valSet, round_idx, sum_metricsList)
            w_globalPre = copy.deepcopy(w_global)

            sum_metrics = sum(sum_metricsList) / len(sum_metricsList)

            if sum_metrics > best_metrics:
                if sum_metrics == 1:
                    sum_metrics =0
                best_metrics = sum_metrics

                saveName = 'best_val_' + str(round_idx + 1) + '.pth.tar'
                save_mode_path = os.path.join(self.args.save_path, saveName)
                state = {
                    'epoch': round_idx + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, save_mode_path)

                logging.info(('best_val——————Average Acc: {r[0]:.4f}\n'.format(r=val_result.avg)))
            else:
                logging.info(('curVal——————Average Acc: {r[0]:.4f}\n'.format(r=val_result.avg)))

            tb_writer.add_scalars('epoch_losses',
                                  {'train_loss': curLoss, 'train_loss_vor': lossVar,
                                   'train_loss_cluster': lossCluster}, round_idx + 1)
        tb_writer.close()

    def _aggregateEMA(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num = training_num + sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] = averaged_params[k] + local_model_params[k] * w
        return averaged_params

    def _aggregatePath(self, w_locals, valSet, round_idx, sum_metricsList):
        training_num = 0
        curDataset = self.client_list[0].split('_')[0].split('-')[0]
        cur_nulNum = {}

        if curDataset == 'MO':
            cur_nulNum = {
                '2':822,
                '4':1713,
                '6':4312,
                'sum': 6847
            }

        elif curDataset == 'TNBC':
            cur_nulNum = {
                '8':603,
                '10':656,
                '11':601,
                'sum':1860
            }

        dyn_initial = 1
        gamma = 0.992
        dynPara = 1 / (dyn_initial * (gamma ** round_idx))
        valWeight = []

        for idx in range(len(w_locals)):
            valWeight.append(sum_metricsList[idx] * dynPara)

        logging.info(valWeight)

        wList = []
        cw_locals = copy.deepcopy(w_locals)
        for idx in range(len(cw_locals)):
            (sample_num, averaged_params, _) = cw_locals[idx]

            wList.append(cur_nulNum[str(sample_num//2)] * math.exp(valWeight[idx]))
            training_num = training_num + wList[idx]
        logging.info(wList)

        vw2 = 0
        for idx in range(0, len(wList)):
            w = wList[idx] / training_num
            wList[idx] = w
            vw2 = vw2 + wList[idx]
        logging.info(vw2)

        (sample_num, averaged_params, _) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params, cmodel = w_locals[i]

                w = wList[i]
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] = averaged_params[k] + local_model_params[k] * w
        return averaged_params

    def get_probmaps2(self, input, input2, model, patch_size, overlap):
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
