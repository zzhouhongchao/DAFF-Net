"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from math import exp
from torch.autograd import Variable
from binary import *


class KL():
    def __init__(self, prior_lambda=10):
        self.prior_lambda = prior_lambda

    def _adj_filt(self, ndims):

        # inner filter 3x3x3
        filt_inner = np.zeros([3] * ndims)
        for j in range(ndims):
            o = [[1]] * ndims
            o[j] = [0, 2]
            filt_inner[np.ix_(*o)] = 1

        # full filter, that makes sure the inner filter is applied 
        filt = np.zeros([ndims, 1] + [3] * ndims)
        for i in range(ndims):
            filt[i,0] = filt_inner
        
        return torch.Tensor(filt)

    def _degree_matrix(self, vol_shape):
        
        ndims = len(vol_shape)
        x = torch.ones([1, ndims, *vol_shape])
        filt = self._adj_filt(ndims)
        
        conv_fn = getattr(F, 'conv%dd' % ndims)
        return conv_fn(x, filt, padding='same', groups=ndims)

    def prec_loss(self, y_pred):

        vol_shape = y_pred.shape[2:]
        ndims = len(vol_shape)
        
        sm = 0
        for i in range(ndims):
            d = i + 2
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = y_pred.permute(r)
            df = y[1:] - y[:-1]
            sm += torch.mean(df * df)
        
        return 0.5 * sm / ndims

    def loss(self, _, y_pred):
        """
        KL loss
        y_pred is assumed to be D*2 channels: first D for mean, next D for logsigma
        D (number of dimensions) should be 1, 2 or 3
        """

        # prepare inputs
        vol_shape = y_pred.shape[2:]
        ndims = len(vol_shape)
        mean = y_pred[:,0:ndims]
        log_sigma = y_pred[:,ndims:]

        # compute the degree matrix
        D = self._degree_matrix(vol_shape).to(y_pred.device)

        # sigma terms
        sigma_term = self.prior_lambda * D * torch.exp(log_sigma) - log_sigma
        sigma_term = torch.mean(sigma_term)

        # precision terms
        prec_term = self.prior_lambda * self.prec_loss(mean)

        return 0.5 * ndims * (sigma_term + prec_term)



def auto_weight_bce(y, y_hat_log):
    with torch.no_grad():
        beta = y.mean(dim=[2, 3], keepdims=True)
    logit_1 = F.logsigmoid(y_hat_log)
    logit_0 = F.logsigmoid(-y_hat_log)
    loss = -(1 - beta) * logit_1 * y \
           - beta * logit_0 * (1 - y)
    return loss.mean()


class edge_loss:
    def __init__(self, class_num = 2):
        self.class_num = class_num
        
    def loss(self, y_true, y_pred):
    
        if torch.equal(y_true, torch.zeros(1).to(y_true.device)):
            y_true = y_pred[0]
            y_pred = y_pred[1]
    
        if torch.equal(y_true, torch.zeros(y_true.shape).to(y_true.device)):
            return torch.mean(y_true)
        
        if torch.equal(y_pred, torch.zeros(y_pred.shape).to(y_pred.device)):
            return torch.mean(y_pred)
    
        if y_true.shape[1] == 1:
            y_true = y_true[:,0].to(torch.int64)
            y_true = F.one_hot(y_true, self.class_num).float()
            y_true = y_true.permute(0,4,1,2,3)
   
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:,0].to(torch.int64)
            y_pred = F.one_hot(y_pred, self.class_num).float()
            y_pred = y_pred.permute(0,4,1,2,3)
            
        y_true = y_true[:,1:]
        y_pred = y_pred[:,1:]
   
        return nn.BCELoss()(y_pred, y_true)
       





def mutil_seg(label):
    label = np.reshape(label,(160,192,160))
    BK = np.zeros((160,192,160))
    CSF = np.zeros((160,192,160))
    GM = np.zeros((160,192,160))
    WM = np.zeros((160,192,160))
    BK[label ==0] =1
    CSF[label==1] =1
    GM[label ==2] =1
    WM[label ==3] =1
    return BK,CSF,GM,WM

def mutil_ASSD(y_true, y_pred):
    T_BK,T_CSF,T_GM,T_WM = mutil_seg(y_true)
    P_BK,P_CSF,P_GM,P_WM = mutil_seg(y_pred)
    a2 = assd(T_CSF,P_CSF)
    a3 = assd(T_GM,P_GM)
    a4 = assd(T_WM,P_WM)
    mean = (a2+a3+a4)/3
    hd1 = [[a2,a3,a4, mean], ]
#    BK = accuracy(T_BK,P_BK)
    return hd1

# 平滑损失
def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])  # dx
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])  # dy
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])  # dz

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def new_gradient_loss(s, penalty='l2'):
    dx = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])  # dx
    dy = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])  # dy
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])  # dz

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return ((d-0.01)*(d-0.01)) / 3.0
 
def new_gradient(s):
    
    dx = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])  # dx
    dy = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])  # dy
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])  # dz
    
    dx = F.pad(dx,(0,0,0,0,0,1))
    dy = F.pad(dy,(0,0,0,1,0,0))
    dz = F.pad(dz,(0,1,0,0,0,0))
    dx = dx ** 2
    dy = dy ** 2
    dz = dz ** 2
    
    gradient_sum = dx + dy + dz
    loss = torch.mean((gradient_sum - 0.01) ** 2)
    return loss



def mse_loss(x, y):
    epsilon = 1e-8
    return torch.mean((x - y) ** 2 + epsilon)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    # 图像维度
    device = I.device
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to(device)   # 指定gpu一致
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)

def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    # Cov(X,Y) = E[(X-E(X))(Y-E(Y))]
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    # 方差
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    D = D1 - D2 + D3
    a = np.minimum(D, 0)
    b = len(a.nonzero()[0]) 
    return b/(1*160*192*160)

# def Get_Ja_hbn(flow):
#     '''
#     Calculate the Jacobian value at each point of the displacement map having
#     size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
#     '''
#     D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
#     D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
#     D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
#     D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
#     D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
#     D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
#     D = D1 - D2 + D3
#     a = np.minimum(D, 0)
#     b = len(a.nonzero()[0]) 
#     return b/(1*192*224*192)


# def NJ_loss(ypred):
#     '''
#     Penalizing locations where Jacobian has negative determinants
#     '''
#     Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
#     return torch.sum(Neg_Jac)

def Get_Ja1(displacement):
    
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])



    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    return D1-D2+D3

def NJ_loss(): 
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    def loss(ypred):

        Neg_Jac = 0.5*(torch.abs(Get_Ja1(ypred)) - Get_Ja1(ypred))
        return torch.sum(Neg_Jac)
    return loss


# def NJ_loss(ypred):
#     Neg_Jac = 0.5*(torch.abs(Get_Ja1(ypred)) - Get_Ja(ypred))
#     return torch.sum(Neg_Jac)

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-




# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob)
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _one_hot_mask_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob)
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-10
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss
    
#     def _dice_mask_loss(self, score, target, mask):
#         target = target.float()
#         mask = mask.float()
#         smooth = 1e-10
#         intersect = torch.sum(score * target * mask)
#         y_sum = torch.sum(target * target * mask)
#         z_sum = torch.sum(score * score * mask)
#         loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss

#     def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        
#         dice = self._dice_loss(inputs, target) 
#         return 1.0 - dice.item()
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()





class calculateDice:
    def __init__(self, class_num = 4):
        self.class_num = class_num
        
    def loss(self, y_true, y_pred):
        #  (1,1)  (1,4)
        if torch.equal(y_true, torch.zeros(1).to(y_true.device)):
            y_true = y_pred[0]
            y_pred = y_pred[1]
    
        if torch.equal(y_true, torch.zeros(y_true.shape).to(y_true.device)):
            return torch.mean(y_true)
        
        if torch.equal(y_pred, torch.zeros(y_pred.shape).to(y_pred.device)):
            return torch.mean(y_pred)
    
        if y_true.shape[1] == 1:
            y_true = y_true[:,0].to(torch.int64)    # 1,1,64,64,64-->1,64,64,64
            y_true = F.one_hot(y_true, self.class_num).float()   # 1,64,64,64,4    # 0和1
            y_true = y_true.permute(0,4,1,2,3)   # 1,4,64,64,64
   
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:,0].to(torch.int64)
            y_pred = F.one_hot(y_pred, self.class_num).float()
            y_pred = y_pred.permute(0,4,1,2,3)
            
        y_true = y_true[:,1:]   # 1,3,64,64,64
        y_pred = y_pred[:,1:]   # 1,3,64,64,64 
   
        return Dice().loss(y_true, y_pred) 
    




    
class FocalDice:
    def __init__(self, class_num = 4):
        self.class_num = class_num
        
    def loss(self, y_true, y_pred):
        #  (1,1)  (1,4)
        if torch.equal(y_true, torch.zeros(1).to(y_true.device)):
            y_true = y_pred[0]
            y_pred = y_pred[1]
    
        if torch.equal(y_true, torch.zeros(y_true.shape).to(y_true.device)):
            return torch.mean(y_true)
        
        if torch.equal(y_pred, torch.zeros(y_pred.shape).to(y_pred.device)):
            return torch.mean(y_pred)
    
        if y_true.shape[1] == 1:
            y_true = y_true[:,0].to(torch.int64)    # 1,1,64,64,64-->1,64,64,64
            y_true = F.one_hot(y_true, self.class_num).float()   # 1,64,64,64,4    # 0和1
            y_true = y_true.permute(0,4,1,2,3)   # 1,4,64,64,64
   
        if y_pred.shape[1] == 1:
            y_pred = y_pred[:,0].to(torch.int64)
            y_pred = F.one_hot(y_pred, self.class_num).float()
            y_pred = y_pred.permute(0,4,1,2,3)
            
        y_true = y_true[:,1:]   # 1,3,64,64,64
        y_pred = y_pred[:,1:]   # 1,3,64,64,64 
   
        return Dice().loss(y_true, y_pred) + Focal().loss(y_true, y_pred)
    
class Dice:
    def __init__(self, epsilon = 1e-5):
        self.epsilon = epsilon

    def loss(self, y_true, y_pred):
    
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims+2))
        
        top = 2 * torch.sum(y_true * y_pred, dim=vol_axes)
        bottom = torch.sum(y_true + y_pred, dim=vol_axes)
        dice = torch.div(top, bottom.clamp(min=self.epsilon))
    
        return -torch.mean(dice)
    
class Focal:
    def __init__(self, alpha=0.25, gamma=2, epsilon = 1e-5):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def loss(self, y_true, y_pred):
    
        y_pred = torch.clamp(y_pred, min=self.epsilon, max=1-self.epsilon) #  把每个值压缩到[min，max]
        logits = torch.log(y_pred / (1 - y_pred))
        weight_a = self.alpha * torch.pow((1 - y_pred), self.gamma) * y_true
        weight_b = (1 - self.alpha) * torch.pow(y_pred, self.gamma) * (1 - y_true)
        loss = torch.log1p(torch.exp(-logits)) * (weight_a + weight_b) + logits * weight_b
    
        return torch.mean(loss)