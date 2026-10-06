# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import numpy as np
import scipy.io as sio
import os

from sklearn import preprocessing


def readData(data_name):
    ''' 读取原始数据和标准类标 '''
    global labels
    path = os.getcwd()+'/data/'+data_name
    if data_name == 'Indian_pines':
        data = sio.loadmat(
            path+'/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat(path+'/Indian_pines_gt.mat')['indian_pines_gt']
    elif data_name == 'PaviaU':
        data = sio.loadmat(path+'/PaviaU.mat')['paviaU']
        labels = sio.loadmat(path+'/PaviaU_gt.mat')['paviaU_gt']
    elif data_name == 'KSC':
        data = sio.loadmat(path+'/KSC.mat')['KSC']
        labels = sio.loadmat(path+'/KSC_gt.mat')['KSC_gt']
    elif data_name == 'Salinas':
        data = sio.loadmat(path+'/Salinas_corrected.mat')['salinas_corrected']
        labels = sio.loadmat(path+'/Salinas_gt.mat')['salinas_gt']
    elif data_name == 'washington':
        data = sio.loadmat(path+'/washington.mat')['washington_datax']
        labels = sio.loadmat(path+'/washington_gt.mat')['washington_gt']
    elif data_name == 'Houston':
        data = sio.loadmat(path+'/Houstondata.mat')['Houstondata']
        labels = sio.loadmat(path+'/Houstonlabel.mat')['Houstonlabel']
    data = np.float64(data)
    labels = np.array(labels).astype(float)
    return data, labels


# def normalizeData(data):
#     ''' 原始数据归一化处理（每条） '''
#     data_norm = np.zeros(np.shape(data))
#     for i in range(np.shape(data)[0]):
#         for j in range(np.shape(data)[1]):
#             data_norm[i, j, :] = preprocessing.normalize(
#                 data[i, j, :].reshape(1, -1))[0]
#     return data_norm

def normalizeData(data):
    ''' 原始数据归一化处理（每个特征） '''
    data_norm = np.zeros(np.shape(data))
    for i in range(np.shape(data)[2]):
        x = preprocessing.normalize(data[:,:,i].reshape([1,-1]))
        data_norm[:,:,i] = x.reshape([data.shape[0],data.shape[1]])
    return data_norm


def selectTrainTestVal(data, labels, pt, pv):
    ''' 从所有类中每类选取训练样本和测试样本
        pt:训练样本比例
        pv：验证样本比例
     '''
    c = int(labels.max())
    x = np.array([], dtype=float).reshape(-1, data.shape[2])  # 训练样本
    xb = []
    x_loc1 = []
    x_loc2 = []
    x_loc = []
    y = np.array([], dtype=float).reshape(-1, data.shape[2])  # 验证
    yb = []
    y_loc1 = []
    y_loc2 = []
    y_loc = []

    z = np.array([], dtype=float).reshape(-1, data.shape[2])  # 测试
    zb = []
    z_loc1 = []
    z_loc2 = []
    z_loc = []

    for i in range(1, c+1):
        #i = 1
        loc1, loc2 = np.where(labels == i)
        # print("isequal",loc1==loc2)
        num = len(loc1)
        order = np.random.permutation(range(num))
        loc1 = loc1[order]
        loc2 = loc2[order]

        num1 = int(np.round(num*(pt)))  # 训练
        num2 = int(np.round(num*(pt+pv)))  # 验证
        # print(num1,num2,num)
        x = np.vstack([x, data[loc1[:num1], loc2[:num1], :]])
        y = np.vstack([y, data[loc1[num1:num2], loc2[num1:num2], :]])
        z = np.vstack([z, data[loc1[num2:], loc2[num2:], :]])

        xb.extend([i]*num1)
        yb.extend([i]*(num2-num1))#caution
        zb.extend([i]*(num-num2))

        x_loc1.extend(loc1[:num1])
        x_loc2.extend(loc2[:num1])

        y_loc1.extend(loc1[num1:num2])
        y_loc2.extend(loc2[num1:num2])

        z_loc1.extend(loc1[num2:])
        z_loc2.extend(loc2[num2:])

        x_loc = np.vstack([x_loc1, x_loc2])
        y_loc = np.vstack([y_loc1, y_loc2])
        z_loc = np.vstack([z_loc1, z_loc2])

        x_loc = np.array(x_loc).transpose()
        y_loc = np.array(y_loc).transpose()
        z_loc = np.array(z_loc).transpose()
    return x, xb, x_loc, y, yb, y_loc,z,zb,z_loc



def selectUlabelSamples(data, labels, pu):
    # x = np.array([], dtype=float).reshape(-1, data.shape[2])  #
    # x_loc1 = []
    # x_loc2 = []
    x_loc = []
    loc1, loc2 = np.where(labels == 0)
    num = len(loc1)
    print(num)
    order = np.random.permutation(range(num))
    loc1 = loc1[order]
    loc2 = loc2[order]
    num1 = int(np.round(num*(pu)))  #
    print(num1)
    x = data[loc1[:num1], loc2[:num1], :]
    x_loc1= loc1[:num1]
    x_loc2= loc2[:num1]
    x_loc = np.vstack([x_loc1, x_loc2])
    x_loc = np.array(x_loc).transpose()
    return x,x_loc


if __name__ == '__main__':
    dataset = ['PaviaU']
    for data_name in dataset:
        data_ori, labels_ori = readData(data_name)
        if data_name == 'KSC':
            pt, pv = 0.05, 0.05
            KSC = np.zeros((512, 614, 24))
            data_ori = np.dstack((data_ori, KSC))
        elif data_name == 'Indian_pines':
            pt, pv = 0.05, 0.05
            Indian_pines = np.zeros((145, 145, 4))
            data_ori = np.dstack((data_ori, Indian_pines))
        elif data_name == 'Houston':
            Houston = np.zeros((1905, 349, 60))
            pt, pv = 0.05, 0.05
            data_ori = np.dstack((data_ori, Houston))
        elif data_name == 'washington':
            pt, pv = 0.05, 0.05
            washington = np.zeros((750, 307, 9))
            data_ori = np.dstack((data_ori, washington))
        elif data_name == 'Salinas':
            pt, pv = 0.05, 0.05
            Salinas = np.zeros((512, 217, 0))
            data_ori = np.dstack((data_ori, Salinas))
        elif data_name == 'PaviaU':
            PaviaU = np.zeros((610, 340, 101))
            pt, pv = 0.03, 0.03
            data_ori = np.dstack((data_ori, PaviaU))

        data_norm = normalizeData(data_ori)
        print('data_norm',data_norm.shape)
        pu = 0.01
  #       selected_bands =[ 144 , 21 , 73 , 61 ,131 ,112, 127, 129 , 23 ,113,  71 , 22 ,128,  33 , 26, 132,  59, 115,
  # 25, 108 , 56 ,109 , 30 , 72 , 27 ,111,  28 , 32,  24 ,110]
  #       selected_bands=[20, 175 ,137 ,174  ,21, 168 ,167,  76, 134 , 69,  32,143 , 33 ,136 , 47 ,170 ,133 , 72,
  #        22 , 36 ,172,  59 , 31 , 27  ,28,  30 , 29 ,  6 , 23 , 25 ,  3 , 26 , 49,   2,  24, 148,
  #        58 ,144  ,10,   5  , 4 , 73, 141, 149 , 71 ,142 ,  7  , 1 , 14 , 13 , 12 , 15,   9, 110,
  #        11 ,  8 ,111 ,107, 108, 109]
        selected_bands=[20, 175
            , 137 ,174  ,21, 168 ,167,  76, 134 , 69,  32,143 , 33 ,136 , 47 ,170 ,133 , 72,
         22 , 36 ,
                        172,  59 , 31 , 27  ,28,  30 , 29 ,  6 , 23 , 25 ,
                        3 , 26 , 49,   2,  24, 148,
         58 ,144  ,10,   5 ,
                        4 , 73, 141, 149 , 71 ,142 ,  7  , 1 , 14 , 13 ,
                        12 , 15,   9, 110,
         11 ,  8 ,111 ,107, 108, 109]
        # 93, 199, 97, 68, 57, 44, 90, 157, 88, 119,
        #  159, 112, 86, 61, 161, 70, 0, 98, 84, 95,
        #  63, 179, 51, 115, 140,135, 34, 185, 113, 166,
        #  66, 89, 186, 40, 150, 181, 81, 139, 195, 117]
         # 147, 60, 163, 156, 123, 192, 77, 106,116, 85,
         # 114, 56, 118, 146, 129, 54, 82, 194, 203, 162]
         # 41, 104, 153, 121, 201, 122, 101, 182, 102, 99,
         # 46,155, 38, 67, 124, 100, 152, 191, 131, 35]
         # 176, 43, 80, 62, 130, 37, 96, 158, 202, 164,
         # 74, 165, 189, 138, 128,78, 50, 151, 160, 178]
         # 17, 105, 125, 83, 65, 200, 180, 48, 132, 193,
         # 42, 120, 52, 64, 18, 19, 91, 45, 126, 177]
         # 127, 198, 16, 183, 94, 184, 79, 171, 188, 92,
         # 154, 53, 197, 103, 187, 39, 87, 55, 75, 169]

        data_ori_new = np.zeros((np.shape(data_norm)[0],np.shape(data_norm)[1],204))
        print('data_ori_new', data_ori_new.shape)
        for i in selected_bands:  # .tolist():
            data_ori_new[:, :, i] = data_norm[:, :, i]
        data_norm = data_ori_new

        train_x, train_y, train_loc, val_x, val_y, val_loc, test_x, test_y, test_loc = selectTrainTestVal(data_norm,
                                                                                                          labels_ori,
                                                                                                          pt, pv)
        Ulab_x, Ulab_loc = selectUlabelSamples(data_norm, labels_ori, pu)

        path = os.getcwd()
        sio.savemat(path + '/data/' + data_name + '/' + data_name + '_pre.mat', {'train_x': train_x,
                                                                                 'train_y': train_y,
                                                                                 'train_loc': train_loc,
                                                                                 'val_x': val_x,
                                                                                 'val_y': val_y, 'val_loc': val_loc,
                                                                                 'test_x': test_x,
                                                                                 'test_y': test_y, 'test_loc': test_loc,
                                                                                 'data_norm': data_norm,
                                                                                 'Ulab_x': Ulab_x, 'Ulab_loc': Ulab_loc,
                                                                                 'labels_ori': labels_ori})
