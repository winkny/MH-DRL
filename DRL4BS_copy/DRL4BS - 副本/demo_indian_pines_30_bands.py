import random
import numpy as np
from DQN import DQN
from HSIDataset import HSIDatsaset
import torch.optim as optim
from itertools import count
import xlwt

import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
import matplotlib.pyplot as plt
from scipy.stats import entropy
import torch
import torch.nn as nn
import time
from sklearn.svm import SVC

import math
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
data_list = ["PaviaU"]
dataset = HSIDatsaset(train_list=data_list, p=0.05)  # modify p
x = dataset.datas[0]

# y_train=dataset.lables[0]
# x_test=dataset.test_sample['pixel']
# y_test=dataset.test_sample['label']
# data = sio.loadmat('data4drl/data_indian_pines_drl.mat')
# print("data", data['x'].shape)
# x = np.float32(data['x'])


### #0. hyper-parameters & data
nb_bands = 103
nb_exp_bands = 30
learning_rate = 0.001
num_episodes = 5000  # train number
state_size = nb_bands
action_size = nb_exp_bands
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
# steps_done = 0
MEMORY_CAPACITY = 2000
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(nb_bands, nb_bands).to(device)
target_net = DQN(nb_bands, nb_bands).to(device)
target_net.load_state_dict(policy_net.state_dict())  # 参数相同
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_func = nn.MSELoss().cuda()


def one_hot(index, n_bands):
    # 波段！
    one_hot_array = np.zeros([n_bands])
    one_hot_array[int(index)] = 1
    return one_hot_array


### step #1. defining the agent (dqn)

class dqn4hsi(object):
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 0.9
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.train_batch = 100
        self.memory_counter = 0  # 记忆库中位值的计数器
        self.memory = np.zeros((MEMORY_CAPACITY, nb_bands * 2 + 3))  # 初始化记忆库

    def step(self, state, action):
        done = 0
        action_new = one_hot(action, state.shape[0])
        s_ = state + action_new
        if np.max(state) == 0:
            reward = self.IEs(s_) * 10
        else:
            reward = self.IEs(s_) * 10 - self.IEs(state) * 10
        if np.sum(s_) == nb_exp_bands:
            done = 1

        return state, action, s_, reward, done

    def IEs(self, state):
        IEs = np.zeros((state.shape[0],))
        for i in range(state.shape[0]):
            if state[i] == 1:
                IEs[i] = entropy(x[:, i])
            else:
                IEs[i] = 0
        IEs = (IEs - np.min(IEs)) / (np.max(IEs) - np.min(IEs))
        IEs = np.sum(IEs) / np.sum(state)
        return IEs

    def act(self, state):  # 确保有10%的概率随机选，而不是只选概率最大的
        if random.random() > self.epsilon:
            action_set = np.squeeze(np.argwhere(state == 0))
            action = random.sample(list(action_set), 1)[0]
            return action
        else:
            invalid_set = np.squeeze(np.argwhere(state > 0))
            state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
            with torch.no_grad():
                prob = policy_net(state)
                prob[0, invalid_set] = -99999  # 这个是为了防止重复选择选过的波段
                return torch.argmax(prob).cpu().data.numpy()

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r], s_, done))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # print("meomory",self.memory_counter)
        if self.memory_counter >= self.train_batch:
            sample_index = np.random.choice(MEMORY_CAPACITY, self.train_batch)
            b_memory = self.memory[sample_index, :]

            # 打包记忆，分开保存进b_s，b_a，b_r，b_s
            b_s = torch.FloatTensor(b_memory[:, :nb_bands]).to(device)
            b_a = torch.LongTensor(b_memory[:, nb_bands:nb_bands + 1].astype(int)).to(device)
            b_r = torch.FloatTensor(b_memory[:, nb_bands + 1:nb_bands + 2]).to(device)
            b_s_ = torch.FloatTensor(b_memory[:, nb_bands + 2:nb_bands + 2 + nb_bands]).to(device)

            # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
            q_eval = policy_net(b_s).gather(1, b_a)  # shape (batch, 1)
            q_next = target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
            q_target = b_r + self.gamma * q_next.max(1)[0]  # shape (batch, 1)
            loss = loss_func(q_eval, q_target)

            # 计算, 更新 eval net
            optimizer.zero_grad()
            loss.backward()  # 误差反向传播
            optimizer.step()
            # print("zhixing!")

            return loss


### step #2. training process
def train(data_list):
    agent = dqn4hsi(state_size, action_size, learning_rate)
    # 下面这部分代码用于加载训练好的模型
    print('===> Try resume from checkpoint')
    if os.path.isdir('checkpoint_' + data_list[0]):
        try:
            checkpoint = torch.load('./checkpoint_' + data_list[0] + '/' + data_list[0] + '.t7')
            policy_net.load_state_dict(checkpoint['state_policy'])  # 从字典中依次读取
            i_episode = checkpoint['i_episode']
            print('===> Load last checkpoint data')
        except FileNotFoundError:
            print('Can\'t found Indian_pines.t7')
    else:
        i_episode = 0
        print('===> Start from scratch')
    print("Start training...")
    RUNNING_R = []
    TARGET_UPDATE = 10
    t1 = time.time()

    while i_episode < num_episodes:
        # print("i_epoch",i_episode)
        ep_r = 0
        state = np.float32(np.zeros((state_size,)))
        for t in count():
            action = agent.act(state)
            # print("action",action)
            pre_state, action, next_state, reward, done = agent.step(state, action)
            # print("done",done)
            ep_r += float(reward)
            agent.store_transition(pre_state, action, reward, next_state, done)
            state = next_state
            agent.learn()
            if done == 1:
                # print("Done")

                if i_episode == 0:

                    # print("Done")
                    if len(RUNNING_R) == 0:
                        RUNNING_R.append(float(ep_r))
                    else:
                        RUNNING_R.append(0.99 * RUNNING_R[-1] + 0.01 * float(ep_r))
                    break
                else:

                    if len(RUNNING_R) == 0:
                        RUNNING_R.append(float(ep_r))
                    else:
                        RUNNING_R.append(0.99 * RUNNING_R[-1] + 0.01 * float(ep_r))
                    break

                # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if i_episode % 20 == 0:
            # print("ep ", i_episode, "/", num_episodes)
            print(
                "Ep:", i_episode,
                "| Ep_r: %.2f" % RUNNING_R[-1],
                # "| loss:%.3f"%(info),
                "| %d bands" % (np.sum(state)),
            )
        if i_episode % 2000 == 0:
            plt.plot(RUNNING_R)
            plt.show()
        i_episode = i_episode + 1
    t2 = time.time()
    print("训练时间", t2 - t1)
    # 保存模型#######################################
    state = {
        'state_policy': policy_net.state_dict(),
        'i_episode': i_episode  # 将epoch一并保存
    }
    if not os.path.isdir('checkpoint_' + data_list[0]):
        os.mkdir('checkpoint_' + data_list[0])
    torch.save(state, './checkpoint_' + data_list[0] + '/' + data_list[0] + '.t7')
    print('model saved')
    #####################################################


### step #3. inference
def test():
    data_list = ["PaviaU"]
    dataset = HSIDatsaset(train_list=data_list, p=0.03)
    x_train = dataset.datas[0]
    y_train = dataset.lables[0]
    x_test = dataset.test_sample['pixel']
    y_test = dataset.test_sample['label']
    selected_bands = np.zeros((nb_exp_bands,))
    state = np.float32(np.zeros((state_size,)))
    for t in range(nb_exp_bands):
        invalid_set = np.squeeze(np.argwhere(state > 0))
        state_new = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        with torch.no_grad():
            prob = policy_net(state_new)
            prob[0, invalid_set] = -99999
            action = torch.argmax(prob).cpu().data.numpy()
        selected_bands[t] = action
        state[action] = 1

        print("{}/{}".format(t, nb_exp_bands))

    print(selected_bands)

    # sio.savemat('results/drl_30_bands_indian_pines.mat', {'selected_bands': selected_bands})
    x_train = np.squeeze(x_train[:, selected_bands.astype(int)])  # array
    print(x_train.shape)
    x_test = np.squeeze(x_test[:, selected_bands.astype(int)])  # array
    acc = 0
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
    t3 = time.time()
    clf = SVC(gamma=gamma_end, kernel='rbf', C=C_end, probability=False, verbose=False)  # 两个参数，手动指定
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    t4 = time.time()

    oa = accuracy_score(y_test, pred)
    per_class_acc = recall_score(y_test, pred, average=None)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_test, pred)
    print("测试时间：", t4 - t3)

    return ([oa, aa, kappa] + per_class_acc.tolist() + [t4 - t3])


##save##########

if __name__ == "__main__":
    result_end = []
    print("train start:")
    train(data_list)

    for i in range(5):
        results = test()
        result_end.append(results)
    #############excel##################
    result_end = np.array(result_end)
    avarage = result_end.mean(axis=0) * 100
    std = result_end.std(axis=0) * 100
    avg = ["%.1f±%.1f" % (avarage[i], std[i]) for i in range(avarage.shape[0])]
    # 创建一个Workbook对象，相当于创建了一个Excel文件
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    sheet = book.add_sheet('Houston', cell_overwrite_ok=True)
    clo_0 = ["OA", "AA", "Kappa"] + list(range(16))
    row_o = ['Houston'] + list(range(5)) + ["avg"]
    for i in range(len(row_o)):
        sheet.write(0, i, row_o[i])
    for j in range(len(clo_0)):
        sheet.write(j + 1, 0, clo_0[j])
    for i in range(len(result_end)):
        for j in range(len(result_end[0])):
            sheet.write(j + 1, i + 1, result_end[i][j])
    for j in range(len(result_end[0])):
        sheet.write(j + 1, 6, avg[j])
    book.save('Houston' + "_ms_svm_" + avg[0] + '.xls')
