
from processing_library import load_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
import time
import os
import numpy as np
data_name='PaviaU'
result_end=[]
for i in range(5):
    os.system('python get_test_prefile.py')
    data_norm, labels_ori, y_train, x_train, train_loc, y_test, x_test, test_loc, y_val, x_val, val_loc, ulab_loc = load_data(
        data_name)
#SVM调参

    # gammas = [0.1,1,10,100,1000,10000]
    # Cs = [0.1,1,10,100,1000,10000]
    # test_acc_init = 0
    # for i in gammas:
    #     for j in Cs:
    #         clf =SVC(gamma=i,kernel='rbf', C=j, probability=False, verbose=False)#两个参数，手动指定
    #         clf.fit(x_train, y_train)
    #         train_acc = clf.score(x_train, y_train)
    #         test_acc = clf.score(x_test, y_test)
    #         print("train_acc",train_acc)
    #         print("test_acc",test_acc)
    #         print(i, j)
    #         if test_acc> test_acc_init:
    #             test_acc_init = test_acc
    #             gamma = i
    #             C=j
    # print(gamma,C)
    t3 = time.time()
    clf =SVC(gamma=10000,kernel='rbf', C=100, probability=False, verbose=False)#两个参数，手动指定
    clf.fit(x_train, y_train)
    print("training accuary:",clf.score(x_train, y_train))
    print("test accuary:",clf.score(x_test, y_test))
    pred= clf.predict(x_test)
    t4 = time.time()
    oa=accuracy_score(y_test,pred)
    per_class_acc=recall_score(y_test,pred,average=None)
    aa=np.mean(per_class_acc)
    kappa=cohen_kappa_score(y_test,pred)
    results = [oa,aa,kappa]
    print(results)
    result_end.append(results)
import xlwt

result_end = np.array(result_end)
avarage = result_end.mean(axis=0) * 100
std = result_end.std(axis=0) * 100
avg = ["%.1f±%.1f" % (avarage[i], std[i]) for i in range(avarage.shape[0])]
# 创建一个Workbook对象，相当于创建了一个Excel文件
book = xlwt.Workbook(encoding="utf-8", style_compression=0)
sheet = book.add_sheet(data_name, cell_overwrite_ok=True)
clo_0 = ["OA", "AA", "Kappa"] + list(range(16))
row_o = [data_name] + list(range(5)) + ["avg"]
for i in range(len(row_o)):
    sheet.write(0, i, row_o[i])
for j in range(len(clo_0)):
    sheet.write(j + 1, 0, clo_0[j])
for i in range(len(result_end)):
    for j in range(len(result_end[0])):
        sheet.write(j + 1, i + 1, result_end[i][j])
for j in range(len(result_end[0])):
    sheet.write(j + 1, 6, avg[j])
book.save(data_name + "_BSnets" + avg[0] + '.xls')


