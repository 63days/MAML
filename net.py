import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from collections import OrderedDict
import math
from torch.nn import init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w_1 = nn.Parameter(torch.DoubleTensor(40, 1))
        self.b_1 = nn.Parameter(torch.DoubleTensor(40))
        self.w_2 = nn.Parameter(torch.DoubleTensor(40, 40))
        self.b_2 = nn.Parameter(torch.DoubleTensor(40))
        self.w_3 = nn.Parameter(torch.DoubleTensor(1, 40))
        self.b_3 = nn.Parameter(torch.DoubleTensor(1))
        self.reset_parameters()
        self.Param = OrderedDict()
        self.set_Param()

        self.mse = nn.MSELoss(reduction='sum')

    def reset_parameters(self):
        init.kaiming_uniform_(self.w_1)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_1)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b_1, -bound, bound)

        init.kaiming_uniform_(self.w_2)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_2)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b_2, -bound, bound)

        init.kaiming_uniform_(self.w_3)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_3)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b_3, -bound, bound)


    def forward(self, input, output, vars=None):
        '''
        :param x:
        :param Param(OrderedDict): {'W1', 'B1' ...}
        :return:
        '''
        if vars is None:
            pred = F.relu(F.linear(input, self.Param['W1'], self.Param['B1']))
            pred = F.relu(F.linear(pred, self.Param['W2'], self.Param['B2']))
            pred = F.linear(pred, self.Param['W3'], self.Param['B3'])
            loss = torch.sqrt(self.mse(output, pred))
        else:
            pred = F.relu(F.linear(input, vars['W1'], vars['B1']))
            pred = F.relu(F.linear(pred, vars['W2'], vars['B2']))
            pred = F.linear(pred, vars['W3'], vars['B3'])
            loss = torch.sqrt(self.mse(pred, output))
        return loss, pred

    def set_Param(self, vars=None):
        if vars is None:
            self.Param['W1'] = self.w_1
            self.Param['B1'] = self.b_1
            self.Param['W2'] = self.w_2
            self.Param['B2'] = self.b_2
            self.Param['W3'] = self.w_3
            self.Param['B3'] = self.b_3
        else:
            self.Param = vars
            self.w_1 = vars['W1']
            self.b_1 = vars['B1']
            self.w_2 = vars['W2']
            self.b_2 = vars['B2']
            self.w_3 = vars['W3']
            self.b_3 = vars['B3']


