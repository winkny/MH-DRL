import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from evalute_net import *
import random
import pandas as pd


def hot2idx(inp):
    return [idx for idx, item in enumerate(inp) if item > 0]


def idx2hot(inp, num_features):
    return [1 if i in inp else 0 for i in range(num_features)]


def KBestAdvisor(pre_action_list, cur_action_list, X_train, Y_train):
    N_feature = X_train.shape[1]
    cur_select_index = hot2idx(cur_action_list)

    pre_select_index = hot2idx(pre_action_list)
    pre_deselect_index = [i for i in range(N_feature) if i not in pre_select_index]

    # assertive selection
    select_select_index = set(cur_select_index) & set(pre_select_index)
    # hesitant selection
    deselect_select_index = set(cur_select_index) & set(pre_deselect_index)

    k = int(len(select_select_index) / 2) + len(deselect_select_index)
    k_action_list = SelectKBest(score_func=mutual_info_classif, k=k)\
        .fit(X_train.iloc[:, cur_action_list == 1], Y_train).get_support(indices=False)
    k_index = [cur_select_index[i] for i, item in enumerate(k_action_list) if item]

    k_assure_index = [i for i in deselect_select_index if i in k_index]
    advise_index = list(select_select_index) + k_assure_index

    return np.array(idx2hot(advise_index, N_feature))


def WrapperAdvisor(pre_action_list, cur_action_list, X_train, Y_train, data_norm, labels_ori, y_test, test_loc):
    N_feature = X_train.shape[1]
    cur_select_index = hot2idx(cur_action_list)

    pre_select_index = hot2idx(pre_action_list)
    pre_deselect_index = [i for i in range(N_feature) if i not in pre_select_index]

    # assertive selection
    select_select_index = set(cur_select_index) & set(pre_select_index)
    # hesitant selection
    deselect_select_index = set(cur_select_index) & set(pre_deselect_index)

    k = int(len(select_select_index) / 2) + len(deselect_select_index)

    best_acc = 0
    best_index = []
    for i in range(5):
        new_index = random.sample(cur_select_index, k)
        select_band = np.array(idx2hot(new_index, N_feature))
        ret = RET(select_band, data_norm, labels_ori, y_test, test_loc)
        acc = ret.acc
        if acc > best_acc:
            best_acc = acc
            best_index = new_index
    k_assure_index = [i for i in deselect_select_index if i in best_index]
    advise_index = list(select_select_index) + k_assure_index
    return np.array(idx2hot(advise_index, N_feature))


# Recursive Feature Elimination
def RFEAdvisor(pre_action_list, cur_action_list, X_train, Y_train, data_norm, labels_ori, y_test, test_loc):
    N_feature = X_train.shape[1]
    cur_select_index = hot2idx(cur_action_list)

    pre_select_index = hot2idx(pre_action_list)
    pre_deselect_index = [i for i in range(N_feature) if i not in pre_select_index]

    # assertive selection
    select_select_index = set(cur_select_index) & set(pre_select_index)
    # hesitant selection
    deselect_select_index = set(cur_select_index) & set(pre_deselect_index)

    k = int(len(select_select_index) / 2) + len(deselect_select_index)
    model = LogisticRegression()
    # model = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=model, n_features_to_select=k)
    rfe = rfe.fit(X_train.iloc[:, cur_action_list == 1], Y_train)
    k_index = [cur_select_index[i] for i, item in enumerate(rfe.support_) if item]
    k_assure_index = [i for i in deselect_select_index if i in k_index]
    advise_index = list(select_select_index) + k_assure_index

    return np.array(idx2hot(advise_index, N_feature))


# RandomForestClassifier
def RFCAdvisor(pre_action_list, cur_action_list, X_train, Y_train, data_norm, labels_ori, y_test, test_loc):
    N_feature = X_train.shape[1]
    cur_select_index = hot2idx(cur_action_list)

    pre_select_index = hot2idx(pre_action_list)
    pre_deselect_index = [i for i in range(N_feature) if i not in pre_select_index]

    # assertive selection
    select_select_index = set(cur_select_index) & set(pre_select_index)
    # hesitant selection
    deselect_select_index = set(cur_select_index) & set(pre_deselect_index)

    k = int(len(select_select_index) / 2) + len(deselect_select_index)
    score = feature_importance(cur_action_list, data_norm, labels_ori, y_test, test_loc)
    score_idx = np.argsort(-score)
    k_index = [cur_select_index[item] for i, item in enumerate(score_idx) if i < k]
    k_assure_index = [i for i in deselect_select_index if i in k_index]
    advise_index = list(select_select_index) + k_assure_index

    return np.array(idx2hot(advise_index, N_feature))


def MIXAdvisor(pre_action_list, cur_action_list, X_train, Y_train, step, MEMORY_UNIT,
               data_norm, labels_ori, y_test, test_loc):
    if step <= MEMORY_UNIT / 3:
        return KBestAdvisor(pre_action_list, cur_action_list, X_train, Y_train)
    elif MEMORY_UNIT / 3 < step < MEMORY_UNIT * 2 / 3:
        return RFEAdvisor(pre_action_list, cur_action_list, X_train, Y_train, data_norm, labels_ori, y_test, test_loc)
    else:
        return RFCAdvisor(pre_action_list, cur_action_list, X_train, Y_train, data_norm, labels_ori, y_test, test_loc)