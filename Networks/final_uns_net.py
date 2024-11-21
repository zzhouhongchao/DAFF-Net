# 配准+分割。  
# 对于两条分支解码器，不单单返回一个分割结果，同时把每一层的特征返回，在配准解码的时候加到里边
# 在配准解码过程中设计一个特征融合模块
# 特征融合模块不改变通道数，模块后使用卷积模块改变通道数
# 特征融合模块： 将5个特征分为S,R,Fuse三类，为这三个分别算一个global权重。然后三个特征分别乘各自的权重进行融合得到initial
# 对initial ， 再计算一下local权重，这个之前是使用的空间注意力。92.23%.  现在可以对initial再去计算一下高低频信息。

import sys
import math
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
from einops.layers.torch import Rearrange


# network
class Net(nn.Module):
    
    def __init__(self, 
                 in_channels: int = 1, 
                 enc_channels: int = 8, 
                 for_train: bool = False,
                 use_checkpoint: bool = True
                ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.for_train = for_train
        self.Encoder = encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint = use_checkpoint
                                   )
        self.Reg_Decoder = Registration_decoder(in_channels=in_channels,
                                     channel_num=enc_channels,
                                     use_checkpoint = use_checkpoint
                                    )
        self.Mor_Decoder = Morphology_decoder(in_channels=in_channels,
                                     channel_num=enc_channels,
                                     use_checkpoint=use_checkpoint 
                                     )
        if self.for_train:
            self.STN = SpatialTransformer_block(mode='bilinear')
            self.NearSTN = SpatialTransformer_block(mode='nearest')
    def forward(self,moving,fixed):

        fix_enc = self.Encoder(fixed)
        mov_enc = self.Encoder(moving)
        
        mseg_feat = self.Mor_Decoder(mov_enc)
        fseg_feat = self.Mor_Decoder(fix_enc)
        
        flow_1 = self.Reg_Decoder(mov_enc,fix_enc,fseg_feat,mseg_feat)
        if self.for_train:
            warped1 = self.STN(moving, flow_1)
            return flow_1,warped1
        else:
            return flow_1,mseg_feat,fseg_feat
  
# encoder  
class encoder(nn.Module):
    
    def __init__(self, 
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = True
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_1 = Conv_block(in_channels, channel_num*2,use_checkpoint)
        self.conv_2 = Conv_block(channel_num*2, channel_num*4,use_checkpoint)
        self.conv_3 = Conv_block(channel_num*4, channel_num*4,use_checkpoint)
        self.conv_4 = Conv_block(channel_num*4, channel_num*8,use_checkpoint)
        self.conv_5 = Conv_block(channel_num*8, channel_num*8,use_checkpoint)
        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x_in):

        x_1 = self.conv_1(x_in)  # 1,1,160,,->1,16,160,,
        
        x = self.downsample(x_1)  # 1,16,80,,
        x_2 = self.conv_2(x)    # 1,32,80,,
        
        x = self.downsample(x_2)  # 1,32,40,,
        x_3 = self.conv_3(x)     # 1,32,40,,
        
        x = self.downsample(x_3)   # 1,32,20,,
        x_4 = self.conv_4(x)     # 1,64,20,,
        
        x = self.downsample(x_4)   # 1，64，10，，
        x_5 = self.conv_5(x)    # 1,64,10,,
        
        return [x_1, x_2, x_3, x_4, x_5]
    
    
class Morphology_decoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = True
               ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_1 = Conv_block(channel_num*8+channel_num*8, channel_num*8,use_checkpoint)  
        self.conv_2 = Conv_block(channel_num*8+channel_num*4, channel_num*4,use_checkpoint)
        self.conv_3 = Conv_block(channel_num*4+channel_num*4, channel_num*4,use_checkpoint)
        self.conv_4 = Conv_block(channel_num*4+channel_num*2, channel_num*2,use_checkpoint)
        
        
       
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x_enc):
        y = self.upsample(x_enc[-1])    # 1,64,10,,,->1,64,20,,,
       
        
        y = torch.cat([y,x_enc[-2]],dim=1)   # 1,128,20,,
        y1 = self.conv_1(y)   # 1,64,20,,
        
        y = self.upsample(y1)   # 1,64,40,,
       
        
        y = torch.cat([y,x_enc[-3]],dim=1)  # 1,96,40,,
        y2 = self.conv_2(y)   # 1,32,40,,
        
        y = self.upsample(y2)   # 1,32,80,,
       
        
        y = torch.cat([y,x_enc[-4]],dim=1)   # 1,64,80,,
        y3 = self.conv_3(y)   # 1,32,80,,,
        
        y = self.upsample(y3)   # 1,32,160,,
        y = torch.cat([y,x_enc[-5]],dim=1)   # 1,48,160,,
        y4 = self.conv_4(y)   # 1,16,160,,
        
        feat = [y1, y2, y3, y4]
        return feat

class Registration_decoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = True
                 ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv_5 = Conv_block(channel_num*8+channel_num*8, channel_num*8,use_checkpoint)
        self.conv_4 = Conv_block(channel_num*8, channel_num*4,use_checkpoint)
        self.conv_3 = Conv_block(channel_num*4, channel_num*4,use_checkpoint)
        self.conv_2 = Conv_block(channel_num*4, channel_num*2,use_checkpoint)
        self.conv_1 = Conv_block(channel_num*2, channel_num*2,use_checkpoint)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2,mode='trilinear')
        self.STN = SpatialTransformer_block(mode='bilinear')
        
        self.Sample = Sample_block()
        self.Integration = Integration_block(int_steps=7)
        self.reghead_5 = RegHead_block(channel_num*8, use_checkpoint)
        self.reghead_4 = RegHead_block(channel_num*4, use_checkpoint)
        self.reghead_3 = RegHead_block(channel_num*4, use_checkpoint)
        self.reghead_2 = RegHead_block(channel_num*2, use_checkpoint)
        self.reghead_1 = RegHead_block(channel_num*2, use_checkpoint)
        
        
        self.att4 = att_registration(channel_num*8,use_checkpoint)
        self.att3 = att_registration(channel_num*4,use_checkpoint)
        self.att2 = att_registration(channel_num*4,use_checkpoint)
        self.att1 = att_registration(channel_num*2,use_checkpoint)
        
   
        
        
    def forward(self,m_enc,f_enc,fseg_feat,mseg_feat):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = f_enc
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = m_enc
        
        # step1
        x5 = torch.cat([x_fix_5,x_mov_5],dim=1)   # 1,128,10,,,

        y5 = self.conv_5(x5)   # 1,64,10,,
        svf_mean_5,svf_sigma_5 = self.reghead_5(y5)
        SVF_5 = self.Sample(svf_mean_5, svf_sigma_5)
        flow_5 = self.Integration(SVF_5) # 1,3,10,
     
        # step2
        flow_5_up = self.ResizeTransformer(flow_5)  # 1,3,20,,
        x_mov_4 = self.STN(x_mov_4,flow_5_up)   # 1,64,20,,
        y = self.upsample(y5)  # 1,64,20,,
          
        y4 = self.att4(y,x_fix_4,x_mov_4,fseg_feat[0],mseg_feat[0])    # 64,64.64.64,64
        y4 = self.conv_4(y4)
        
        
        svf_mean_4,svf_sigma_4 = self.reghead_4(y4)
        SVF_4 = self.Sample(svf_mean_4, svf_sigma_4)
        x = self.Integration(SVF_4) # 1,3,10,
        
        flow_4 = x + self.STN(flow_5_up,x)
        
        # step3
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.STN(x_mov_3,flow_4_up)
        y = self.upsample(y4)   # 1,32,40,,
        
        y3  = self.att3(y,x_fix_3,x_mov_3,fseg_feat[1],mseg_feat[1])
        y3 = self.conv_3(y3)
        
        svf_mean_3,svf_sigma_3 = self.reghead_3(y3)
        SVF_3 = self.Sample(svf_mean_3, svf_sigma_3)
        x = self.Integration(SVF_3) # 1,3,10,
        flow_3 = x + self.STN(flow_4_up,x)
        
        # step4
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.STN(x_mov_2,flow_3_up)
        y = self.upsample(y3)   # 1,32,80,,
        
        y2 = self.att2(y,x_fix_2,x_mov_2,fseg_feat[2],mseg_feat[2])
        y2 = self.conv_2(y2)
                
        svf_mean_2,svf_sigma_2 = self.reghead_2(y2)
        SVF_2 = self.Sample(svf_mean_2, svf_sigma_2)
        x = self.Integration(SVF_2) # 1,3,10,
        flow_2 = x + self.STN(flow_3_up,x)
        # step5
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.STN(x_mov_1,flow_2_up)
        y = self.upsample(y2)  # 1,16,160,,
       
        y1 = self.att1(y,x_fix_1,x_mov_1,fseg_feat[3],mseg_feat[3])
        y1 = self.conv_1(y1)
        
        svf_mean_1,svf_sigma_1 = self.reghead_1(y1)
        SVF_1 = self.Sample(svf_mean_1, svf_sigma_1)
        x = self.Integration(SVF_1) # 1,3,10,
        
        flow_1 = x + self.STN(flow_2_up,x)
        
        return flow_1
        
     
class Sample_block(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, mean, sigma):

        noise = torch.normal(torch.zeros(mean.shape), torch.ones(mean.shape)).to(mean.device)   # torch.normal()从给定均值和方差的正态分布中抽取随机数
        x_out = mean + torch.exp(sigma/2.0) * noise    
        
        return x_out
    
class Integration_block(nn.Module):
    
    def __init__(self, int_steps=7):
        super().__init__()
        self.int_steps = int_steps

    def forward(self, Velocity):
        shape = Velocity.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        # grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(Velocity.device)
        
        flow = Velocity / (2.0 ** self.int_steps)
        for _ in range(self.int_steps):
            new_locs = grid + flow
            for i in range(len(shape)):
                new_locs[:,i] = 2*(new_locs[:,i]/(shape[i]-1) - 0.5)

            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]
            flow = flow + nnf.grid_sample(flow, new_locs, align_corners=True, mode='bilinear')

        return flow
    
class RegHead_block(nn.Module):
    
    def __init__(self, 
                 in_channels: int, 
                 use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.Conv = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1,padding='same')
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Norm = nn.InstanceNorm3d(in_channels)
        
        self.mean_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1,padding='same')
        self.mean_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.mean_head.weight.shape))
        self.mean_head.bias = nn.Parameter(torch.zeros(self.mean_head.bias.shape))
 
        self.sigma_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1,padding='same')
        self.sigma_head.weight = nn.Parameter(Normal(0, 1e-10).sample(self.sigma_head.weight.shape))
        self.sigma_head.bias = nn.Parameter(torch.full(self.sigma_head.bias.shape, -10.0))
    
    def Reg_forward(self, x_in):
        
        x = self.Conv(x_in)
        x = self.LeakyReLU(x)
        x = self.Norm(x)   # F9fuse
        
        Velocity_mean = self.mean_head(x)
        Velocity_sigma = self.sigma_head(x)
        
        return Velocity_mean, Velocity_sigma
    
    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            Velocity_mean, Velocity_sigma = checkpoint.checkpoint(self.Reg_forward, x_in)
        else:
            Velocity_mean, Velocity_sigma = self.Reg_forward(x_in)
        
        return Velocity_mean, Velocity_sigma     
    
# 空间转换网络 STN
class SpatialTransformer_block(nn.Module):
    
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        # grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:,i] = 2*(new_locs[:,i]/(shape[i]-1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2,1,0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
    
    
class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x




class Conv_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernal_size=3, 
                 stride=1, 
                 padding='same', 
                 alpha=0.2,
                 use_checkpoint: bool=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernal_size, stride, padding)
        
        self.acti = nn.LeakyReLU(alpha)
        self.Norm = nn.InstanceNorm3d(out_channels)
       
    def InsNormActivate(self,x):
        x = self.acti(x)
        x = self.Norm(x)
        return x
        
    def Conv_forward(self, x_in):
        out = self.conv1(x_in)
        out = self.InsNormActivate(out)
        out = self.conv2(out)
        out = self.InsNormActivate(out)
        return out
    def forward(self,x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)
        
        return x_out


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
    
class SE_block(nn.Module):
    def __init__(self, channel):
        super(SE_block, self).__init__()
       
        self.avg_pool = nn.AdaptiveAvgPool3d(1)   
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // 3, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // 3, channel, bias=False),
                nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y
    
class SegHead_block(nn.Module):
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.seg_head = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)
    
    def Seg_forward(self, x_in):
        
        x = self.seg_head(x_in)
        x_out = self.softmax(x)
        
        return x_out
    
    def forward(self, x_in):
        
        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Seg_forward, x_in)
        else:
            x_out = self.Seg_forward(x_in)
        
        return x_out
    
##############################
###########att################
# gaussian 1
class att_registration(nn.Module):
    def __init__(self,
                 channel,
                 use_checkpoint: bool = True):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.conv_R = nn.Conv3d(channel*2,channel,kernel_size=3,stride=1,padding='same')
        self.conv_S = nn.Conv3d(channel*2,channel,kernel_size=3,stride=1,padding='same')
        self.global_weight = AdaptiveParameterLearning(3,channel)
        self.conv = nn.Conv3d(channel*3, channel, kernel_size=3, stride=1, padding='same')
        self.acti_fn = nn.LeakyReLU(0.2)
        self.Norm = nn.InstanceNorm3d(channel)    
        self.spatial_att = SpatialAttention()
        self.GaussianFilter = GaussianKernel(channels=channel)
        self.conv_out = nn.Conv3d(channel*2, channel, kernel_size=3,stride=1,padding='same')
        
    def fuse_forward(self,Fuse,R_f,R_m,S_f,S_m):   
        
        R  = torch.cat([R_f,R_m],dim=1)
        R = self.conv_R(R)
        R = self.acti_fn(R)
        R = self.Norm(R)
        
        S = torch.cat([S_f,S_m],dim=1)
        S =self.conv_S(S)
        S = self.acti_fn(S)
        S = self.Norm(S)
        
        features = [R,S,Fuse]
        global_weights = self.global_weight(features)
        x = torch.cat([R*global_weights[0,0],S*global_weights[0,1],Fuse*global_weights[0,2]],dim=1)
        x = self.conv(x)
        x = self.acti_fn(x)
        x = self.Norm(x)
        
        low_frequency = self.GaussianFilter(x)
        high_frequency = x-low_frequency
        
        low_frequency_out = self.spatial_att(low_frequency)
        high_frequency_out = self.spatial_att(high_frequency)
        
        out = torch.cat([low_frequency_out,high_frequency_out],dim=1)
        out = self.conv_out(out)
        out = self.acti_fn(out)
        out = self.Norm(out)
        
        return out
    def forward(self,Fuse,R_f,R_m,S_f,S_m):
        if self.use_checkpoint and Fuse.requires_grad:
            x_out = checkpoint.checkpoint(self.fuse_forward, Fuse,R_f,R_m,S_f,S_m)
        else:
            x_out = self.fuse_forward(Fuse,R_f,R_m,S_f,S_m)
        return x_out

class AdaptiveParameterLearning(nn.Module):
    def __init__(self, num_features, num_channels):
        super(AdaptiveParameterLearning, self).__init__()
        self.num_features = num_features
        # 为每个特征图的全局均值和最大值预测权重
        self.fc = nn.Sequential(
            nn.Linear(num_channels * num_features * 2,num_channels * num_features),  # 假设输入特征的维度是通道数的两倍（均值和最大值）
            nn.ReLU(),
            nn.Linear(num_channels * num_features, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels , num_features),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        # features 应该是一个列表，包含五个特征图
        batch_size = features[0].size(0)
        global_stats = []
        
        for f in features:
            # 全局平均池化和全局最大池化
            avg_pool = nnf.adaptive_avg_pool3d(f, 1)
            max_pool = nnf.adaptive_max_pool3d(f, 1)
            # 将均值和最大值展平并合并
            pooled_features = torch.cat((avg_pool, max_pool), dim=1).view(batch_size, -1)
            global_stats.append(pooled_features)
        
        # 将所有特征图的统计信息合并
        global_stats = torch.cat(global_stats, dim=1)
        
        # 通过全连接层计算权重
        weights = self.fc(global_stats)
        
        return weights


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_3 = nn.Conv3d(2, 1, kernel_size=3, padding='same')
        self.conv_5 = nn.Conv3d(2, 1, kernel_size=5, padding='same')
        self.conv_7 = nn.Conv3d(2, 1, kernel_size=7, padding='same')
        self.conv_out = nn.Conv3d(3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_avg = torch.mean(x,dim=1,keepdim=True)
        x_max,_ = torch.max(x,dim=1,keepdim=True)
        out = torch.cat([x_avg,x_max],dim=1)
        
        out_3 = self.conv_3(out)
        out_5 = self.conv_5(out)
        out_7 = self.conv_7(out)
        
        out = torch.cat([out_3,out_5,out_7],dim=1)
        out = self.conv_out(out)
        out = self.sigmoid(out)
        
        return x * out  # 应用空间注意力图

    
class GaussianKernel(nn.Module):
    def __init__(self, channels, kernel_size=3, sigma=1):
        super(GaussianKernel, self).__init__()
       
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size, kernel_size).view(kernel_size, kernel_size, kernel_size).float()
        y_grid = x_grid.transpose(0, 1)
        z_grid = x_grid.transpose(0, 2)
        xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.
        
        gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(
                              -torch.sum((xyz_grid - mean)**2., dim=-1) / 
                              (2*variance)
                          )
       
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1, 1)
        
        self.register_buffer('weight', gaussian_kernel)
        self.groups = channels
        self.kernel_size = kernel_size

    def forward(self, x):
        
        padding = self.kernel_size // 2
        return nnf.conv3d(x, self.weight, padding=padding, groups=self.groups)