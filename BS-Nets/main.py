# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:17:28 2019

@author: ld
"""

import torch.optim as optim
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import time

import numpy as np

# import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from processing_library import load_data, one_hot, disorder, next_batch, next_batch_unl
from processing_library import contrary_one_hot, expend, windowFeature, pca_trans_expend
from processing_library import save_result, random_cut_bands
import copy
from processing_library import index_band_selection

from NET import BSNET_Conv

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


###############################################################################
# data_norm,labels_ori,x_train,y_train,train_loc,x_test,y_test,test_loc=load_data('Indian_pines')
class RET:
    def __init__(self, data_name="KSC"):
        self.data_name = data_name
        # self.band_num = band_num

        data_norm, labels_ori, y_train, x_train, train_loc, y_test, x_test, test_loc, y_val, x_val, val_loc, ulab_loc = load_data(
            self.data_name)
        nrows, ncols, ndim = data_norm.shape
        dim_input = ndim

        ulab_loc = val_loc  # caution!
        # dim_input = ndim
        batch_size = 512
        display_step = 100
        step = 1
        index = batch_size

        lr_intit = 0.001  # 初始学习率【调参】
        lr_decay_step = 300  # 每过lr_decay_step个step，学习就下降0.9
        lr_decay_rate = 0.9
        num_classification = int(np.max(labels_ori))  # 类别数
        w = 5  # 图像块大小

        num_epoch = 500  # 训练循环次数

        pca_dum_out = 5

        data_norm = expend(data_norm, w)

        # data2_norm = expend(data2_norm,w)

        dim_input = np.shape(data_norm)[2]

        data_norm = torch.tensor(data_norm, dtype=torch.float32).cuda()
        #        train_loc = torch.tensor(train_loc, dtype=torch.int32).cuda()
        # Y_train = torch.tensor(Y_train, dtype=torch.long).cuda()
        ulab_loc = torch.tensor(ulab_loc, dtype=torch.int32).cuda()
        # print("ulab_loc",ulab_loc.shape)
        ###############################################################################
        newdata = torch.zeros(
            [1024, w, w, data_norm.shape[2]]).cuda()

        def _windowFeature_torch(data_expand, loc, w, newdata=newdata):
            newdata = torch.zeros_like(newdata)
            for i in range(loc.shape[0]):
                x1 = loc[i, 0]  # 没错！
                y1 = loc[i, 1]
                x2 = loc[i, 0] + w
                y2 = loc[i, 1] + w
                c = data_expand[x1:x2, y1:y2, :]
                newdata[i, :, :, :] = c
            return newdata[:loc.shape[0], :, :, :]

        t1 = time.time()
        net = BSNET_Conv(ndim)
        # net = torch.nn.DataParallel(net)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(net.parameters(), lr=lr_intit, momentum=0.9)
        # optimizer = optim.Adam(net.parameters(), lr=lr_intit)

        #############training###################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(device)
        net.to(device)
        # net = torch.nn.DataParallel(net, [0,1])

        epoch = 0
        step = 1
        running_loss = 0.0
        time_train_start = time.clock()
        index = batch_size

        print("Start training...")
        start = time.time()
        while epoch < num_epoch:
            if epoch == 1:
                t = time.time()
                print('一轮时间' + str(t - start))
            with torch.no_grad():
                batch_unl_loc = next_batch_unl(ulab_loc, index, batch_size)
                batch_x_unl = _windowFeature_torch(data_norm, batch_unl_loc, w).permute(
                    0, 3, 1, 2)
            optimizer.zero_grad()
            #   print("batch_x_unl",batch_x_unl.shape)
            outputs_unl, BRW = net(batch_x_unl)
            #    print(BRW.shape)
            # print(outputs_unl.shape)
            loss = criterion(outputs_unl, batch_x_unl) + 1e-12 * torch.sum(BRW)
            loss.backward()
            optimizer.step()  # 更新分类层权值
            running_loss += loss.item()
            if step % display_step == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.7f' %
                      (epoch + 1, step + 1, running_loss / 100),
                      " | learning_rate: %.5f" % (optimizer.param_groups[0]["lr"]))
                running_loss = 0.0
            if step % lr_decay_step == 0:
                # 学习率衰减
                for p in optimizer.param_groups:
                    p['lr'] *= lr_decay_rate

            index = index + batch_size
            step += 1
            #            print(step)
            if index > ulab_loc.shape[0]:
                index = batch_size
                #                print('epoch:',epoch)
                epoch = epoch + 1

                perm0 = np.arange(len(ulab_loc))
                np.random.shuffle(perm0)
                ulab_loc = ulab_loc[perm0]
                # Start next epoch
        time_train_end = time.clock()
        print('Finished Training')
        t2 = time.time()
        # save model
        self.bandscore = np.sum(BRW.detach().cpu().numpy(), axis=0).reshape([-1])
        # bands_idx = np.argsort(np.abs(self.bandscore))[-self.band_num:]
        # print(bands_idx)
        # x_train_cut = x_train[:,bands_idx]
        # x_test_cut = x_test[:,bands_idx]
        # #SVM调参
        # gammas = [0.1,1,10,100,1000,10000]
        # Cs = [0.1,1,10,100,1000,10000]
        # test_acc_init = 0
        # for i in gammas:
        #     for j in Cs:
        #         clf =SVC(gamma=i,kernel='rbf', C=j, probability=False, verbose=False)#两个参数，手动指定
        #         clf.fit(x_train_cut, y_train)
        #         train_acc = clf.score(x_train_cut, y_train)
        #         test_acc = clf.score(x_test_cut, y_test)
        #         if test_acc> test_acc_init:
        #             test_acc_init = test_acc
        #             gamma = i
        #             C=j
        # t3 = time.time()
        # clf =SVC(gamma=gamma,kernel='rbf', C=C, probability=False, verbose=False)#两个参数，手动指定
        # clf.fit(x_train_cut, y_train)
        # print("training accuary:",clf.score(x_train_cut, y_train))
        # print("test accuary:",clf.score(x_test_cut, y_test))
        # pred= clf.predict(x_test_cut)
        # t4 = time.time()
        # oa=accuracy_score(y_test,pred)
        # per_class_acc=recall_score(y_test,pred,average=None)
        # aa=np.mean(per_class_acc)
        # kappa=cohen_kappa_score(y_test,pred)
        # self.results = [oa,aa,kappa]+per_class_acc.tolist()+[t2-t1,t4-t3]


if __name__ == '__main__':
    ret1 = RET("KSC")
    bandscore1 = ret1.bandscore
    ret2 = RET("washington")
    bandscore2 = ret2.bandscore
    ret3 = RET("Salinas")
    bandscore3 = ret3.bandscore
    bandscore = (bandscore1 + bandscore2 + bandscore3) / 3
    band_num = 60
    bands_idx = np.argsort(np.abs(bandscore))[-band_num:]
    print(bands_idx)

    # print(ret.results)
