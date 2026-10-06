import numpy as np
import scipy.io as sio
import os

from sklearn import preprocessing

def labels_to_idxs(labels):
    return np.stack(np.where(labels != 0)).transpose()

def random_cut_bands(data):
    '''
    按均匀分布随机选择波段集合
    '''
    
#    num_bands = data.shape[1]
#    for i in range(data.shape[0]):
#        num_zero = num_bands - int(np.random.uniform(low=1, high=num_bands))
#        index_zero_bands = np.random.choice(range(num_bands),num_zero,replace=False)
#        data[i,index_zero_bands,:,:] = 0
    num_bands = data.shape[1]
    num_zero = num_bands - int(np.random.uniform(low=1, high=num_bands))
    index_zero_bands = np.random.choice(range(num_bands),num_zero,replace=False)
    data[:,index_zero_bands,:,:] = 0
    return data

def read_data(data_name):
    ''' 读取原始数据和标准类标 '''
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
        labels = sio.loadmat(path+'/washington_gt.mat')["washington_gt"]
    elif data_name == 'Houston':
        data = sio.loadmat(path+'/Houstondata.mat')['Houstondata']
        labels = sio.loadmat(path+'/Houstonlabel.mat')['Houstonlabel']
    else:
        raise NotImplementedError
    data = np.float64(data)
    labels = np.array(labels).astype(float)
    idxs = labels_to_idxs(labels)
    #return data, labels,idxs
    return data, labels,np.where(labels != 0)

def normalize_data(data):
    ''' 原始数据归一化处理（每个特征） '''
    data_norm = np.zeros(np.shape(data))
    for i in range(np.shape(data)[2]):
        x = preprocessing.normalize(data[:,:,i].reshape([1,-1]))
        data_norm[:,:,i] = x.reshape([data.shape[0],data.shape[1]])
    return data_norm

if __name__=="__main__":
    data,label,idx=read_data("KSC")
    print("data",data.shape)
    for i in range (data.shape[2]):
        a=min(data[:,:,i].flatten())
        print(a)


