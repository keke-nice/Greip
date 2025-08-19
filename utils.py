""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import argparse
from torch.autograd import Variable

def get_args():
    parser = argparse.ArgumentParser(description='Train ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 训练参数
    parser.add_argument('-g', '--GPU', dest='GPU', type=str, default='0',
                        help='the index of GPU')
    parser.add_argument('-p', '--pp', dest='num_workers', type=int, default=8,
                        help='num_workers')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=64*64*64,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-fn', '--fold_num', type=int, default=5,
                        help='fold_num', dest='fold_num')
    parser.add_argument('-fi', '--fold_index', type=int, default=0,
                        help='fold_index:0-fold_num', dest='fold_index')
    parser.add_argument('-rT', '--reTrain', dest='reTrain', type=int, default=0,
                        help='Load model')
    parser.add_argument('-rD', '--reData', dest='reData', type=int, default=0,
                        help='re Data')

    # 图片参数
    parser.add_argument('-f', '--form', dest='form', type=str, default='Resize',
                        help='the form of input img')
    parser.add_argument('-wg', '--weight', dest='weight', type=int, default=36,
                        help='the weight of img')
    parser.add_argument('-hg', '--height', dest='height', type=int, default=36,
                        help='the height of img')
    parser.add_argument('-n', '--frames_num', dest='frames_num', type=int, default=256,
                        help='the num of frames')

    parser.add_argument('-tr', '--temporal_aug_rate', type=float, default=0,
                        help='temporal_aug_rate', dest='temporal_aug_rate')
    parser.add_argument('-sr', '--spatial_aug_rate', type=float, default=0,
                        help='spatial_aug_rate', dest='spatial_aug_rate')
    parser.add_argument('-gr', '--gamma_corr_rate', type=float, default=0,
                        help='gamma_corr_rate', dest='gamma_corr_rate')
    parser.add_argument('-qr', '--quantized_rate', type=float, default=0,
                        help='quantized_rate', dest='quantized_rate')
    parser.add_argument('-cr', '--cubic_spli_rate', type=float, default=0,
                        help='cubic_spli_rate', dest='cubic_spli_rate')
    parser.add_argument('-lr', '--light_aug_rate', type=float, default=0,
                        help='light_aug_rate', dest='light_aug_rate')
    
    parser.add_argument('-gm1', '--gamma1', type=float, default=2.2,
                        help='gamma1', dest='gamma1')
    parser.add_argument('-gm2', '--gamma2', type=float, default=2.2,
                        help='gamma2', dest='gamma2')
    parser.add_argument('-gart', '--gart', type=float, default=0,
                        help='gamama_rate', dest='gamma_rate')
    parser.add_argument('-la', '--la', type=float, default=0,
                        help='the ratio of light', dest='l_a')
    parser.add_argument('-cs', '--cs', type=float, default=0,
                        help='down sample size', dest='c_size')
    parser.add_argument('-qs', '--qs', type=float, default=0,
                        help='quantity size', dest='q_size')
    parser.add_argument('-tl', '--tl', type=float, default=0,
    help='quantity size', dest='t_len')

    parser.add_argument('-bb', '--bb', type=float, default=1.0,
                        help='bb', dest='b')
    parser.add_argument('-v', '--v', dest='version', type=int, default=0,
                        help='version')
    return parser.parse_args()

def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr-HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp))/len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr))*(HR_rel - np.mean(HR_rel))) / (
                0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class P_loss3(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, gt_lable, pre_lable):
        M, N, A = gt_lable.shape
        gt_lable = gt_lable - torch.mean(gt_lable, dim=2).view(M, N, 1)
        pre_lable = pre_lable - torch.mean(pre_lable, dim=2).view(M, N, 1)
        aPow = torch.sqrt(torch.sum(torch.mul(gt_lable, gt_lable), dim=2))
        bPow = torch.sqrt(torch.sum(torch.mul(pre_lable, pre_lable), dim=2))
        pearson = torch.sum(torch.mul(gt_lable, pre_lable), dim=2) / (aPow * bPow + 0.01)
        loss = 1 - torch.sum(torch.sum(pearson, dim=1), dim=0)/(gt_lable.shape[0] * gt_lable.shape[1])
        return loss


class SP_loss(nn.Module):
    def __init__(self, device, clip_length=256, delta=3, loss_type=1, use_wave=False):
        super(SP_loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = clip_length
        self.device = device
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype = torch.float).to(self.device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        self.two_pi_n = two_pi_n.to(self.device)
        self.hanning = hanning.to(self.device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, pred = None, flag = None):  # all variable operation
        fps = 30

        hr = gt.clone() 

        hr[hr.ge(self.high_bound)] = self.high_bound-1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps
        preds = wave * self.hanning

        preds = preds.view(batch_size, 1, -1)
        f_t = f_t.repeat(batch_size, 1).view(batch_size, -1, 1)

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)

        complex_absolute = torch.sum(preds * torch.sin(f_t*tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t*tmp), dim=-1) ** 2

        target = hr - self.low_bound
        target = target.type(torch.long).view(batch_size)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound



        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            norm_t = (torch.ones(batch_size).to(self.device) / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            loss = self.cross_entropy(complex_absolute, target)

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1;

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx