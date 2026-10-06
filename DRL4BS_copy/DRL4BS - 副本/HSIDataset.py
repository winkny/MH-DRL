# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:34:50 2021

@author: ld
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import read_data, normalize_data,random_cut_bands
import sys
import random
#import sklearn

from sklearn import model_selection

class HSIDatsaset():

    def __init__(self, train_list=['Indian_pines'],p=0.05, root_dir="."):
        """
        Args:
        train_list:训练数据集名称，list
        test_list:测试数据，list
        root_fir:数据文件夹，str
        p:训练样本比例
            
        """
        # if len(train_list)!=1:
        #     raise  Exception
        self.data_norm=[]
        self.datas = []
        self.lables = []
        self.full_datas=[]
        self.full_labels=[]
        self.loc_full=[]
        # self.labeled_idxs=[]
        for i in range(len(train_list)):
            _data, _label,_idx = read_data(train_list[i])
            _data = normalize_data(_data)
            x_data = _data[_idx[0],_idx[1]]#取出有标签样本
            y_data = _label[_idx[0],_idx[1]]
            #划分训练集和测试集
            X_train, X_test, y_train, y_test =model_selection.train_test_split(x_data,y_data,test_size=1-p, shuffle=True)
            self.data_norm.append(_data)
            self.loc_full.append(_idx)
            self.datas.append(X_train)
            self.lables.append(y_train)
            self.full_datas.append(x_data)
            self.full_labels.append(y_data)
            self.test_sample ={'pixel': X_test, 'label': y_test}
            # self.labeled_idxs.append(_idx)
            
            #划分训练与测试样本
            
            
            
            
#        self.__getitem__([1,2,3])

    def __len__(self):
        if self.is_train:
            return 1000
        else:
            return 128 #TODO

    # def test_batch(self):
    #     return self.test_sample
        
        
    def next_batch(self, batch_size=64):
       
        data_id = random.choice(list(range(len(self.datas)))) #随机选择数据集
#        print(self.datas[0].shape)
        #取数据与标签
        batch_data_id = np.random.choice(list(range(self.datas[data_id].shape[0])),batch_size)
        
        data = self.datas[data_id][batch_data_id]
#        print(data.shape)
        label = self.lables[data_id][batch_data_id]
        
        #随机采样
#        num_bands = data.shape[1]
#        selected_band = int(np.random.uniform(low=50, high=num_bands))
#        idx_bands = np.random.choice(range(num_bands),selected_band,replace=False)
#        
#        data_cut = data[:,idx_bands]
        
#
        sample = {'pixel': data, 'label': label}
#        print(sample['pixel'].shape,sample['label'].shape)
#
#
        return sample
    
if __name__ == '__main__':
    HSIs = HSIDatsaset()
    for i in range(10):
        sample = HSIs.next_batch(128)
        print(i, sample['pixel'].shape, sample['label'].shape)    
#    dataloader = DataLoader(HSIs, batch_size=4,
#                        shuffle=True, num_workers=4)
#    
#    for i_batch, sample_batched in enumerate(dataloader):
#        print(i_batch, sample_batched['pixel'].size(),
#              sample_batched['label'].size())
    