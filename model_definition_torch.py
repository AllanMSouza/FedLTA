import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import copy
import random
import numpy as np
import sys
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

batch_size = 16

class LocalModel(nn.Module):
    def     __init__(self, base, head):
        super(LocalModel, self).__init__()

        self.base = base
        self.head = head
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out
# ====================================================================================================================
class DNN_proto_4(nn.Module):
    def __init__(self, input_shape=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN_proto_4, self).__init__()

        self.fc0 = nn.Linear(input_shape, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        rep = F.relu(self.fc2(x))
        x = self.fc(rep)
        output = F.log_softmax(x, dim=1)
        return output, rep
# ====================================================================================================================
class DNN_proto_2(nn.Module):
    def __init__(self, input_shape=1 * 28 * 28, mid_dim=100, num_classes=10):
        try:
            super(DNN_proto_2, self).__init__()

            self.fc0 = nn.Linear(input_shape, mid_dim)
            self.fc = nn.Linear(mid_dim, num_classes)
        except Exception as e:
            print("DNN_proto_2")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def forward(self, x):
        try:
            x = torch.flatten(x, 1)
            rep = F.relu(self.fc0(x))
            x = self.fc(rep)
            output = F.log_softmax(x, dim=1)
            return output, rep
        except Exception as e:
            print("DNN_proto_2 forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================
class DNN(nn.Module):
    def __init__(self, input_shape=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
# ====================================================================================================================
class CNN(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(input_shape, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            # self.fc1 = nn.Linear(256, 120)
            self.fc1 = nn.Linear(mid_dim, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        except Exception as e:
            print("CNN")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.pool(F.relu(self.conv1(x)))
            out = self.pool(F.relu(self.conv2(out)))
            rep = torch.flatten(out, 1)  # flatten all dimensions except batch
            out = F.relu(self.fc1(rep))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
            return out
        except Exception as e:
            print("CNN forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
# ====================================================================================================================
class CNN_proto(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_proto, self).__init__()
            self.conv1 = nn.Conv2d(input_shape, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(mid_dim, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)
        except Exception as e:
            print("CNN proto")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.pool(F.relu(self.conv1(x)))
            out = self.pool(F.relu(self.conv2(out)))
            rep = torch.flatten(out, 1)  # flatten all dimensions except batch
            out = F.relu(self.fc1(rep))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
            return out, rep
        except Exception as e:
            print("CNN proto forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================
class Logistic(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
# ====================================================================================================================