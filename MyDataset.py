# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image
from numpy.fft import fft, ifft, rfft, irfft
from torch.autograd import Variable
import random
from scipy import interpolate

def transform(image):
    image = transF.resize(image, size=(300, 600))
    image = transF.to_tensor(image)
    image = transF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

class Data_VIPL(Dataset):
    def __init__(self, root_dir, frames_num, args, transform = None):
        self.root_dir = root_dir
        self.frames_num = int(frames_num)
        self.datalist = os.listdir(root_dir)
        self.datalist = sorted(self.datalist)
        self.num = len(self.datalist)
        self.transform = transform
        self.args=args

    def __len__(self):
        return self.num

    def getLabel(self, nowPath, Step_Index):
        bvp_name = 'Label_CSI/BVP_Filt.mat'
        bvp_path = os.path.join(nowPath, bvp_name)
        bvp = scio.loadmat(bvp_path)['BVP']
        bvp = np.array(bvp.astype('float32')).reshape(-1)
        bvp = bvp[Step_Index:Step_Index + self.frames_num]
        bvp = (bvp - np.min(bvp))/(np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')

        gt_name = 'Label_CSI/HR.mat'
        gt_path = os.path.join(nowPath, gt_name)
        gt = scio.loadmat(gt_path)['HR']
        gt = np.array(gt.astype('float32')).reshape(-1)
        gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])
        gt = gt.astype('float32')
        return gt, bvp

    def __getitem__(self, idx):
        idx = idx
        img_name = 'STMap'
        STMap_name = 'STMap_YUV_Align_CSI.png'
        nowPath = os.path.join(self.root_dir, self.datalist[idx])
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])
        Step_Index = int(temp['Step_Index'])
        gt, bvp = self.getLabel(nowPath, Step_Index)
        STMap_Path = os.path.join(nowPath, img_name)
        feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name))
        # feature_map = feature_map[:, Step_Index:Step_Index + self.frames_num, :]
        With, Max_frame, _ = feature_map.shape
        map_ori = feature_map[:, Step_Index:Step_Index + self.frames_num, :]

        Step_Index_aug = Step_Index
        map_aug = map_ori
        aug_ratio = (1.0*random.uniform(0, 100)/100.0)
        if aug_ratio < self.args.spatial_aug_rate:
          if self.args.spatial_aug_rate > 0:
                temp_ratio = (1.0*random.uniform(0, 100)/100.0)
                Index = np.arange(With)
                if temp_ratio < 0.3: #0.3的概率打乱两行
                    Index[random.randint(0, With-1)] = random.randint(0, With-1)
                    Index[random.randint(0, With-1)] = random.randint(0, With-1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.6:
                    Index[random.randint(0, With-1)] = random.randint(0, With-1)
                    Index[random.randint(0, With-1)] = random.randint(0, With-1)
                    Index[random.randint(0, With-1)] = random.randint(0, With-1)
                    Index[random.randint(0, With-1)] = random.randint(0, With-1)
                    map_aug = map_ori[Index]
                elif temp_ratio < 0.9:
                    np.random.shuffle(Index[random.randint(0, With-1):random.randint(0, With-1)])#随机选择一个切片随机打乱，起始值大于终止值则返回为空
                    map_aug = map_ori[Index]
                else:
                    np.random.shuffle(Index)
                    map_aug = map_ori[Index]
                  
    
        elif aug_ratio < (self.args.spatial_aug_rate+self.args.cubic_spli_rate):
          if self.args.cubic_spli_rate > 0:#随机下采样到128 最好的结果是210
                indices = np.sort(np.random.choice(range(self.frames_num), int(self.args.c_size), replace=False))
                if not np.isin([0], indices):
                    indices = np.concatenate(([0], indices))
                if not np.isin([self.frames_num-1], indices):
                    indices = np.concatenate((indices, [self.frames_num-1]))
                map_sampled = map_ori[:, indices, :]
                x = np.arange(self.frames_num)
                f = interpolate.interp1d(indices, map_sampled, kind='cubic', 
                                         #fill_value="extrapolate", 
                                         axis=1)
                map_aug = f(x)
        elif aug_ratio < (self.args.spatial_aug_rate+self.args.cubic_spli_rate+self.args.gamma_corr_rate):
            if self.args.gamma_corr_rate > 0:#gamma=2.2
                    if self.args.gamma2==0:
                        gamma = random.randint(8,23)/10
                        map_aug = self.args.gamma_rate*map_aug + (1-self.args.gamma_rate)*np.power(map_ori, gamma)
                    else: #gamma2=0.8 rate=0.7
                        map_aug = self.args.gamma_rate*map_aug + (1-self.args.gamma_rate)*np.power(map_aug, self.args.gamma2)
                  
                    gamma_corr_flag = 1
        elif aug_ratio < (self.args.spatial_aug_rate+self.args.cubic_spli_rate+self.args.gamma_corr_rate+self.args.light_aug_rate):
            if self.args.light_aug_rate > 0:#args.l_a设置为0.6
                  b_mat = np.array([[1,0,0],[0,1,0],[0,0,1]])
                  mat = np.random.rand(3, 3)-0.5
                  mat = (self.args.l_a)*b_mat + (1-self.args.l_a)*mat
                  # mat /= np.sum(mat, axis=0)
                  arr_2d = map_ori.reshape(-1, 3)
                  mat_2d = mat.reshape(3, -1)
                  map_aug_2d = np.dot(arr_2d, mat_2d)
                  map_aug = map_aug_2d.reshape(With, self.frames_num, 3)

        elif aug_ratio < (self.args.spatial_aug_rate+self.args.cubic_spli_rate+self.args.gamma_corr_rate+self.args.light_aug_rate+self.args.quantized_rate):
            if self.args.quantized_rate > 0:#[800,1000]
                  map_aug = np.round(map_ori*self.args.q_size)/self.args.q_size
                  quantized_flag = 1
        else:
          if self.args.temporal_aug_rate > 0:  
              if Step_Index + self.frames_num + int(self.args.t_len) < Max_frame:
                      Step_Index_aug = int(random.uniform(0, int(self.args.t_len-1)) + Step_Index)
                      map_aug=feature_map[:, Step_Index_aug:Step_Index_aug + self.frames_num, :]
              else:
                  map_aug = map_ori

        
        gt_aug, bvp_aug = self.getLabel(nowPath, Step_Index_aug)
        for c in range(map_ori.shape[2]):
            for r in range(map_ori.shape[0]):
                map_ori[r, :, c] = 255 * ((map_ori[r, :, c] - np.min(map_ori[r, :, c])) / (0.00001 +
                            np.max(map_ori[r, :, c]) - np.min(map_ori[r, :, c])))
        for c in range(map_aug.shape[2]):
            for r in range(map_aug.shape[0]):
                map_aug[r, :, c] = 255 * ((map_aug[r, :, c] - np.min(map_aug[r, :, c])) / \
                                (0.00001 +np.max(map_aug[ r, :,c]) - np.min(map_aug[r, :, c])))
        map_ori = Image.fromarray(np.uint8(map_ori))
        map_aug = Image.fromarray(np.uint8(map_aug))

        map_ori = self.transform(map_ori)
        map_aug = self.transform(map_aug)
        return (map_ori, bvp, gt, map_aug, bvp_aug, gt_aug)
      

def CrossValidation(root_dir, fold_num=5,fold_index=0):
    datalist = os.listdir(root_dir)
    # datalist.sort(key=lambda x: int(x))
    num = len(datalist)
    test_num = round(((num/fold_num) - 2))
    train_num = num - test_num
    test_index = datalist[fold_index*test_num:fold_index*test_num + test_num-1]
    train_index = datalist[0:fold_index*test_num] + datalist[fold_index*test_num + test_num:]
    return test_index, train_index

def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for sub_file in filesList:
        now = os.path.join(root_path, sub_file)
        img_path = os.path.join(now, os.path.join('STMap', Pic_path))
        temp = cv2.imread(img_path)
        Num = temp.shape[1]
        Res = Num - frames_num - 1  # 可能是Diff数据
        Step_num = int(Res/Step)
        for i in range(Step_num):
            Step_Index = i*Step
            temp_path = sub_file + '_' + str(1000 + i) + '_.mat'
            scio.savemat(os.path.join(save_path, temp_path), {'Path': now, 'Step_Index': Step_Index})
            Index_path.append(temp_path)
    return Index_path

