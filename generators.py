import os
import glob
import sys
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import random


'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class DatasetOASIS1_35label(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
      
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]   # (1, 160, 192, 160)
        Start = self.files[index].rfind('/O')
        trainsegDir = '../data/OASIS1_35label/4label' + self.files[index][Start:-12]
        img_seg = sitk.GetArrayFromImage(sitk.ReadImage(trainsegDir + '_fseg.nii.gz'))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr,img_seg,self.files[index]




class newDatasetlearn2reg(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
    
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]  
        img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr)-np.min(img_arr))
        img_arr = img_arr[:,16:176,16:208,:]
        return img_arr,self.files[index]


class Datasetlearn2reg(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
    
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]  
        img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr)-np.min(img_arr))
        
        return img_arr,self.files[index]



class DatasetLPBA_seg(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]   # (1, 160, 192, 160)
        Start = self.files[index].rfind('/S')
        trainsegDir = '../data/LPBA/train_seg' + self.files[index][Start:-7]
        img_seg = sitk.GetArrayFromImage(sitk.ReadImage(trainsegDir + '_seg.nii.gz'))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr,img_seg,self.files[index]


class DatasetOASIS3_seg(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]   # (1, 160, 192, 160)
        Start = self.files[index].rfind('/O')
        trainsegDir = '../data/OASIS3/train_seg' + self.files[index][Start:-7]
        img_seg = sitk.GetArrayFromImage(sitk.ReadImage(trainsegDir + '.nii.gz'))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr,img_seg,self.files[index]


class DatasetOASIS1_seg(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]   # (1, 160, 192, 160)
        Start = self.files[index].rfind('/O')
        trainsegDir = '../data/OASIS1/train_seg' + self.files[index][Start:-7]
        img_seg = sitk.GetArrayFromImage(sitk.ReadImage(trainsegDir + '_fseg.nii.gz'))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr,img_seg,self.files[index]

class Dataset(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]   # (1, 160, 192, 160)
        return img_arr,self.files[index]


    

    
    
