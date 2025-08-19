# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import utils
from torchvision import models
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

np.set_printoptions(threshold=np.inf)
sys.path.append('..')

class adaIN(nn.Module):

    def __init__(self, eps=1e-5):
        super(adaIN, self).__init__()
        self.eps = eps

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = out_in#[B,512,4,16]
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)#[B,512,4,16]
        return out


class ResnetAdaINBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetAdaINBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = adaIN()
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = adaIN()

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(x, gamma, beta)
        out = self.relu1(x)
        out = self.conv2(x)
        out = self.norm2(x, gamma, beta)#[B,512,4,16]
        return x+out
class Encoder_noise(nn.Module):
    def __init__ (self, ):
        super(Encoder_noise, self).__init__()
        model_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1_noise = model_resnet.conv1
        self.bn1_noise = model_resnet.bn1
        self.relu_noise = model_resnet.relu
        self.layer1_noise = model_resnet.layer1#64
        self.layer2_noise = model_resnet.layer2#128
        self.layer3_noise = model_resnet.layer3#256
        self.layer4_noise = model_resnet.layer4#512
    def forward(self, x):
        x2 = self.conv1_noise(x)
        x2 = self.bn1_noise(x2)
        x2 = self.relu_noise(x2)

        x2_1 = self.layer1_noise(x2)#[512,64,32,128]
        x2_2 = self.layer2_noise(x2_1)#[512,128,16,64]
        x2_3 = self.layer3_noise(x2_2)#[512,256,8,32]
        em2 = self.layer4_noise(x2_3)#[512,512,4,16]
        return x2_1, x2_2, x2_3, em2
class BVP_decoder(nn.Module):
    def __init__(self) -> None:
        super(BVP_decoder,self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),#升维[512,512,4,32]
            BasicBlock(512, 256, [2, 1], downsample=1),#[512,256,2,32]
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),#[512,256,2,64]
            BasicBlock(256, 64, [1, 1], downsample=1),#[512,64,2,64]
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),#[512,64,2,128]
            BasicBlock(64, 32, [2, 1], downsample=1),#[512,32,1,128]
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),#[512,32,1,256]
            BasicBlock(32, 1, [1, 1], downsample=1),#[512,1,1,256]
        )
    def forward(self, x):
        x1 = self.up1(x)#[512,256,2,32]
        x1 = self.up2(x1)#[512,64,2,64]
        x1 = self.up3(x1)#[512,32,1,128]
        Sig = self.up4(x1).squeeze(dim=1)#[512,1,256]
        return Sig

class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
                 )
        self.downsample = downsample
        self.Res = Res

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        return F.relu(out)



class BaseNet(nn.Module):
    def __init__(self, dim=512, K=5120, ada_num=2):
        super(BaseNet, self).__init__()
        # model_resnet = models.resnet18(pretrained=True)
        model_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.layer1 = model_resnet.layer1#64
        self.layer2 = model_resnet.layer2#128
        self.layer3 = model_resnet.layer3#256
        self.layer4 = model_resnet.layer4#512
  
        self.K = K
        self.gamma = nn.Linear(512, 512, bias=False)
        self.beta = nn.Linear(512, 512, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

        
        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512)
        )
        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(512) for i in range(ada_num)])
        self.FC = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True)
        )
        self.encoder_noise = Encoder_noise()
        self.classifier = nn.Linear(512,1)
        self.bvp_decoder = BVP_decoder()
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("noise", torch.randn(dim, K))
        self.noise = nn.functional.normalize(self.noise, dim=0)
        self.register_buffer("fusion", torch.randn(dim, K))
        self.fusion = nn.functional.normalize(self.fusion, dim=0)
        self.register_buffer("label_queue", torch.randn(K)+70)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_tgt", torch.randn(dim, 2560))
        self.queue_tgt = nn.functional.normalize(self.queue_tgt, dim=0)
        self.register_buffer("noise_tgt", torch.randn(dim, 2560))
        self.noise_tgt = nn.functional.normalize(self.noise_tgt, dim=0)
        self.register_buffer("fusion_tgt", torch.randn(dim, 2560))
        self.fusion_tgt = nn.functional.normalize(self.fusion_tgt, dim=0)
        self.register_buffer("label_tgt", torch.randn(2560))
        self.register_buffer("tgt_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, noise, fusion, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        self.label_queue[ptr: ptr + batch_size] = labels
        self.noise[:, ptr : ptr + batch_size] = noise.T
        self.fusion[:, ptr : ptr + batch_size] = fusion.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _tgt_dequeue_and_enqueue(self, keys,noise, fusion, labels):
        batch_size = keys.shape[0]
        ptr = int(self.tgt_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_tgt[:, ptr : ptr + batch_size] = keys.T
        self.label_tgt[ptr: ptr + batch_size] = labels
        self.noise_tgt[:, ptr : ptr + batch_size] = noise.T
        self.fusion_tgt[:, ptr : ptr + batch_size] = fusion.T
        ptr = (ptr + batch_size) % 2560  # move pointer

        self.tgt_ptr[0] = ptr

    def get_av(self, x):
        av = torch.mean(torch.mean(x, dim=-1), dim=-1)#[512,64]
        min, _ = torch.min(av, dim=1, keepdim=True)
        max, _ = torch.max(av, dim=1, keepdim=True)
        av = torch.mul((av-min),((max-min).pow(-1)))
        return av
    
    def cal_gamma_beta(self, x1_1, x1_2, x1_3):
        x1_add = x1_1
        x1_add = self.ada_conv1(x1_add)+x1_2
        x1_add = self.ada_conv2(x1_add)+x1_3
        x1_add = self.ada_conv3(x1_add)#[B,256,4,16]

        gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)#[B,256,1,1]
        gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)#[B,512]
        return  gamma, beta
    
    def loss_contineous(self, x, label, label_temperature=0.1):
        diff = torch.abs(label.unsqueeze(1) - self.label_queue.clone().detach()) #[512,k]
        weight = -diff
        weight = torch.nn.functional.softmax(weight/label_temperature, dim=-1)
        sim = torch.matmul(x, self.queue.clone().detach()) #[512,k]
        log_probs = torch.nn.functional.log_softmax(sim, dim=-1)
        loss = -(weight * log_probs).sum(dim=-1).mean()
        if torch.isnan(loss):
            print('There in nan loss found in loss contineous')
        return loss

    def loss_orthogonal(self, x, y): #x [B,dim] y[dim,k]
        xy = torch.mm(x, y)
        x_norm = torch.norm(x, dim=1)
        y_norm = torch.norm(y, dim=0)
        loss = (1/3)*(F.mse_loss(xy, torch.zeros_like(xy)) + F.mse_loss(x_norm, torch.ones_like(x_norm)) + F.mse_loss(y_norm, torch.ones_like(y_norm)))
        if torch.sum(torch.isnan(loss)) > 0:
          print('There in nan loss found in loss orthogonal')
        return loss
        
    def loss_dissimilarity(self, feature, fusion):
        tensor1_normalized = torch.nn.functional.normalize(feature, p=2, dim=1)
        tensor2_normalized = torch.nn.functional.normalize(fusion, p=2, dim=1)
        cosine_similarity = torch.sum(torch.mul(tensor1_normalized, tensor2_normalized), dim=1)
        loss = torch.sum(torch.pow(cosine_similarity + 1, 2))/(feature.shape[0]*fusion.shape[0])
        if torch.sum(torch.isnan(loss)) > 0:
          print('There in nan loss found in loss dissimilarity')
        return loss
    
    def forward(self, x1, label, mode='train'):
        x2 = x1
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x1 = self.layer1(x1)#[512,64,32,128]
        av1_1 = self.get_av(x1)#[512,64]
        x1 = self.layer2(x1)#[512,128,16,64]
        av1_2 = self.get_av(x1)#[512,128]
        x1 = self.layer3(x1)#[512,256,8,32]
        av1_3 = self.get_av(x1)#[512,256]
        em1 = self.layer4(x1)#[512,512,4,16]
        av1_4 = self.get_av(em1)#[512,512]
        av1_4 = nn.functional.normalize(av1_4, dim=1)
        #noise
        x2_1, x2_2, x2_3, em2 = self.encoder_noise(x2)
        av2_4 = self.get_av(em2)#[512,512]
        av2_4 = nn.functional.normalize(av2_4, dim=1)
      

        gamma, beta = self.cal_gamma_beta(x2_1, x2_2, x2_3)
        for i in range(len(self.adaIN_layers)):
            fea_fusion = self.adaIN_layers[i](em1, gamma, beta)#[B,512,4,16]
        HR_fusion = self.classifier(self.avgpool(fea_fusion).view(x1.size(0), -1))
        Sig_fusion = self.bvp_decoder(fea_fusion)

        fea_fusion_av = self.get_av(fea_fusion)#[B,512]
        fea_fusion_av = nn.functional.normalize(fea_fusion_av, dim=1)
        loss_cons = self.loss_contineous(av1_4, label)
        # loss_ori_1 = self.loss_orthogonal(av1_4, self.noise.clone().detach()) 
        loss_ori_2 = self.loss_orthogonal(av2_4, self.queue.clone().detach()) 
        loss_ori = loss_ori_2
        loss_diss = self.loss_dissimilarity(fea_fusion_av, av1_4)
        if mode=='train':
            self._dequeue_and_enqueue(av1_4, av2_4, fea_fusion_av, label)
        elif mode=='eval':
            self._tgt_dequeue_and_enqueue(av1_4, av2_4, fea_fusion_av,label)
        return Sig_fusion, HR_fusion, loss_cons, loss_ori
        
        
        