# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:38:39 2018

@author: Jiantong Chen
"""

import copy
import os
import random
import numpy as np
import scipy
import torch
import scipy.io as sio
from sklearn.decomposition import PCA

'''def label_RGB(label):
    colors = np.array([[0, 0, 0],
                       [255, 255, 0],
                       [0, 128, 0],
                       [0, 255, 0],
                       [255, 0, 0],
                       [0, 0, 255],
                       [0, 255, 255],
                       [255, 0, 255]])
    img = np.zeros([label.shape[0], label.shape[1], 3], dtype=np.uint8)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            img[i, j, :] = colors[label[i, j]]
    print(img.shape)
    return img'''


def load_data(data_name):
    '''读取数据'''
    path = os.getcwd()
    pre = sio.loadmat(path + '/data_new/' + 'Indian_pines_pre.mat')

    data_norm = pre['data_norm']
    labels_ori = pre['labels_ori'].flatten()
    x_train = pre['data_train']
    y_train = pre['label_train'].flatten()
    train_loc = pre['loc_train']
    x_test = pre['data_test']
    y_test = pre['label_test'].flatten()
    test_loc = pre['loc_test']
    x_full = pre['data1']
    y_full = pre['label1'].flatten()
    loc_full = pre['loc1']

    return data_norm, labels_ori, x_train, y_train, train_loc, x_test, y_test, test_loc, x_full, y_full, loc_full


def disorder(X, Y):
    '''打乱顺序'''
    index_train = np.arange(X.shape[0])
    np.random.shuffle(index_train)
    X = X[index_train, :]
    Y = Y[index_train]
    return X, Y


def next_batch(image, lable, index, batch_size):
    '''数据分批'''
    start = index - batch_size
    end = index
    return image[start:end, :], lable[start:end]


def window_Feature(data, loc, w):
    '''从扩展矩阵中得到窗口特征'''
    size = np.shape(data)
    data_expand = np.zeros((int(size[0] + w - 1), int(size[1] + w - 1), size[2]))
    for j in range(size[2]):
        data_expand[:, :, j] = np.lib.pad(data[:, :, j], ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2))),
                                          'symmetric')
    newdata = np.zeros([loc.shape[0], w, w, data_expand.shape[2]])
    for i in range(loc.shape[0]):
        x1 = loc[i, 0]
        y1 = loc[i, 1]
        x2 = loc[i, 0] + w
        y2 = loc[i, 1] + w
        c = data_expand[x1:x2, y1:y2, :]
        newdata[i, :, :, :] = c
    return newdata


def one_hot(lable, class_number):
    '''转变标签形式'''
    one_hot_array = np.zeros([len(lable), class_number])
    for i in range(len(lable)):
        one_hot_array[i, int(lable[i] - 1)] = 1
    return one_hot_array


def contrary_one_hot(label):
    '''将onehot标签转化为真实标签'''
    size = len(label)
    label_ori = np.empty(size)
    for i in range(size):
        label_ori[i] = np.argmax(label[i]) + 1
    return label_ori


def plot(data_norm, y_pr):
    plot_max = np.zeros((data_norm.shape[0], data_norm.shape[1]))
    print("plot_max", plot_max.shape)
    # print(loc_full[0].shape)
    # print('y_pr', y_pr)
    for i in range(plot_max.shape[0]):
        for j in range(plot_max.shape[1]):
            plot_max[i, j] = y_pr[plot_max.shape[1] * i + j]
    return plot_max


def label_RGB(label):
    label = label.astype(int)

    colors = np.array([[0, 0, 0.5156],
                       [0, 0, 0.7656],
                       [0, 0.0156, 1.0000],
                       [0, 0.2656, 1.0000],
                       [0, 0.5156, 1.0000],
                       [0, 0.7656, 1.0000],
                       [0.0156, 1.0000, 0.9844],
                       [0.2656, 1.0000, 0.7344],
                       [0.5156, 1.0000, 0.4844],
                       [0.7656, 1.0000, 0.2344],
                       [1.0000, 0.9844, 0],
                       [1.0000, 0.7344, 0],
                       [1.0000, 0.4844, 0],
                       [1.0000, 0.2344, 0],
                       [0.5451, 0.2706, 0.0745],
                       [0.7344, 0, 0],
                       [0.5, 0, 0],
                       ])
    img = np.zeros([label.shape[0], label.shape[1], 3], dtype=np.uint8)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            img[i, j, :] = colors[label[i, j]]
    print(img.shape)
    return img
