import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from net import Net
import pickle
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Meta(nn.Module):
    def __init__(self, inner_lr=1e-2, outer_lr=1e-3):
        super(Meta, self).__init__()
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.net = Net()
        self.outer_optim = optim.Adam(list(self.net.Param.values()), lr=self.outer_lr)
        # print(list(self.net.Param.values()))

    def forward(self, k_x, k_y, q_x, q_y):
        batch_size = k_x.size(0)
        losses = None
        for i in range(batch_size):
            lossA, predA = self.net(k_x[i], k_y[i])
            # print(lossA)
            grad = torch.autograd.grad(lossA, self.net.Param.values(), create_graph=True, retain_graph=True)
            fast_weights = OrderedDict(
                (name, p - self.inner_lr * g) for ((name, p), g) in zip(self.net.Param.items(), grad))
            # print(fast_weights)

            lossB, predB = self.net(q_x[i], q_y[i], fast_weights)

            if losses is None:
                losses = lossB
            else:
                losses = losses + lossB

        self.outer_optim.zero_grad()
        losses.backward()
        self.outer_optim.step()
        self.net.set_Param()

        return losses.item() / batch_size

    def test(self, file_name='./weights/w.pkl', k=10, q=15):
        self.load_weights(file_name=file_name)
        amplitude = np.random.uniform(0.1, 5.0, size=1)
        phase = np.random.uniform(0., np.pi, size=1)
        xs = np.linspace(-5, 5).reshape(-1, 1)
        ys = amplitude * np.sin(xs + phase)
        idx = np.random.choice(50, k+q, replace=False)
        k_idx, q_idx = idx[:k], idx[k:]

        k_x, k_y, q_x, q_y = torch.from_numpy(xs[k_idx]), torch.from_numpy(ys[k_idx]), torch.from_numpy(
            xs[q_idx]), torch.from_numpy(ys[q_idx])
        k_x, k_y, q_x, q_y = k_x.to(device), k_y.to(device), q_x.to(device), q_y.to(device)
        xs, ys = torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)

        pre_loss, pre_pred = self.net(xs, ys)

        lossA, predA = self.net(k_x, k_y)
        grad = torch.autograd.grad(lossA, self.net.Param.values(), create_graph=True, retain_graph=True)
        fast_weights = OrderedDict(
            (name, p - self.inner_lr * g) for ((name, p), g) in zip(self.net.Param.items(), grad))

        post_loss, post_pred = self.net(xs, ys, fast_weights)

        print('| pre loss: {:.3f} | post loss: {:.3f} |'.format(pre_loss, post_loss))

        xs, ys = xs.cpu().numpy(), ys.cpu().numpy()
        pre_pred, post_pred = pre_pred.cpu().detach().numpy(), post_pred.cpu().detach().numpy()
        f, ax = plt.subplots()
        line1 = ax.plot(xs, ys, label='Ground Truth')
        line2 = ax.plot(xs, pre_pred, label='Pre Update')
        line3 = ax.plot(xs, post_pred, label='One Grad Step')
        ax.legend()
        plt.savefig('graph.png')

    def save_weights(self, file_name='./weights/w.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.net.Param, f)

    def load_weights(self, file_name='./weights/w.pkl'):
        with open(file_name, 'rb') as f:
            P = pickle.load(f)

        self.net.set_Param(P)



