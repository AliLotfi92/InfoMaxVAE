import torch
from torch.autograd import Variable
import numpy as np


class Act_units(object):

    def __init__(self, num_samples=20, sig=0.02):
        self.num_samples = num_samples
        self.sig = sig
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def comp(self, net, input):

        z_list = []
        mean = []

        for i in range(len(input)):
            x_test, _ = iter(input).next()
            x_test = Variable(x_test, requires_grad=False).to(self.device)

            for i in range(self.num_samples):
                _, _, _, z = net(x_test)
                z_list.append(z.squeeze().cpu().data.numpy())

            mean.append(np.mean(z_list, axis=0))

        mean_z = np.concatenate(mean, axis=0)
        mean_xz = np.mean(mean_z, axis=0)
        cov_x = np.mean(np.square(mean_z - mean_xz), axis=0)
        print(cov_x.min())

        return np.sum(cov_x >= 0.05)


