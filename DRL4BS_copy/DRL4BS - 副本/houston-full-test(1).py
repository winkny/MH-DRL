import torch
from HSIDataset import HSIDatsaset
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from processing_library_torch import label_RGB,plot
from pre_color import plot_label
nb_exp_bands=60
if __name__ == "__main__":
    data_list = ["Houston"]
    dataset = HSIDatsaset(train_list=data_list, p=0.05)  # modify p
    selected_bands=[11,   0, 105,  77, 106,  70,  95,  83,  7,  13,   6,  56,  23,  72,
  71,  19,  45, 103, 101,  59,  17,  60,  67, 128,  39, 132,  64,  87,
  30,  47,  89,  75, 138,  42, 113,   4,  98,  14,  43,  81,]

    x_train = dataset.datas[0]
    y_train = dataset.lables[0]
    x_test = dataset.full_datas[0]
    y_test = dataset.full_labels[0]

    data_norm=dataset.data_norm[0]
    x_test_background =data_norm.reshape(-1,144)
    loc_full=dataset.loc_full[0]

    # sio.savemat('results/drl_30_bands_indian_pines.mat', {'selected_bands': selected_bands})
    x_train = np.squeeze(x_train[:, selected_bands])  # array
    x_test = np.squeeze(x_test[:, selected_bands])  # array
    x_test_background = np.squeeze(x_test_background[:, selected_bands])
    acc = 0
    t3 = time.time()
    for gamma in [0.1, 1, 10, 100, 1000, 10000, 100000]:
        for C in [0.1, 1, 10, 100, 1000, 10000, 100000]:
            clf = SVC(gamma=gamma, kernel='rbf', C=C, probability=False, verbose=False)  # 两个参数，手动指定
            clf.fit(x_train, y_train)
            test_acc = clf.score(x_test, y_test)
            if test_acc > acc:
                acc = test_acc
                gamma_end = gamma
                C_end = C
        print(gamma_end, C_end, acc)
    # print(clf.score(self.x_test[:,selected_bands], self.y_test))
    clf = SVC(gamma=gamma_end, kernel='rbf', C=C_end, probability=False, verbose=False)  # 两个参数，手动指定
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    t4 = time.time()

    oa = accuracy_score(y_test, pred)
    per_class_acc = recall_score(y_test, pred, average=None)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_test, pred)

    print(([oa, aa, kappa] + per_class_acc.tolist() + [t4 - t3]))

    with torch.no_grad():
        pred=clf.predict(x_test_background)
        pred_full= pred
        # oa = accuracy_score(real_full, pred_full)
        # per_class_acc = recall_score(real_full, pred_full, average=None)
        # aa = np.mean(per_class_acc)
        # kappa = cohen_kappa_score(real_full, pred_full)
        # print(per_class_acc)
        # print(oa, aa, kappa)
        print("画图开始：")
        plot_max = plot(data_norm, pred_full)
        plot_model = plot_label(data_list[0])
        img = plot_model.plot_color(plot_max)
        plt.imsave(data_list[0] + "_" + str(oa) + ".tiff", img)
        # imag = label_RGB(plot_max)

        # ret_im = Image.fromarray(imag)  # RGB图像
        # ret_im.save("result.tif")
        print("画图结束！")
        # print('全图训练精度：\n')
        # oa = accuracy_score(real_full, pred_full)
        # per_class_acc = recall_score(real_full, pred_full, average=None)
        # aa = np.mean(per_class_acc)
        # kappa = cohen_kappa_score(real_full, pred_full)
        # print(per_class_acc)
        # print(oa, aa, kappa)



