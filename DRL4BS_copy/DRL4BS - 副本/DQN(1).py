import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



#网络定义
class DQN(torch.nn.Module):
    def __init__(self,input_dim,out_dim):
        super(DQN, self).__init__()
        self.f1=nn.Linear(input_dim,400)
        self.f2 =nn.Linear(400, out_dim)

    def forward(self, data):

        x = self.f1(data)
        x = F.relu(x)
        x = self.f2(x)
        return x