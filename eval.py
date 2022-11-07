

import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import numpy as np
import os
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import Active_Units_Comp


warnings.filterwarnings('ignore')


class Classification(nn.Module):
    def __init__(self, z_dim=2):
        super(Classification, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),

        )

    def forward(self, z):
        return self.net(z).squeeze()


class CNNVAE1(nn.Module):

    def __init__(self, z_dim=2):
        super(CNNVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2 * z_dim, 1),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 128, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False, no_dec=False):
        if no_enc:
            gen_z = Variable(torch.randn(100, z_dim, 1, 1),
                             requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

        elif no_dec:
            stats = self.encode(x)
            mu = stats[:, :self.z_dim]
            logvar = stats[:, self.z_dim:]
            z = self.reparametrize(mu, logvar)
            return z.squeeze()

        else:
            stats = self.encode(x)
            mu = stats[:, :self.z_dim]
            logvar = stats[:, self.z_dim:]
            z = self.reparametrize(mu, logvar)
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()


def recon_loss(x_recon, x):
    n = x.size(0)
    loss = F.binary_cross_entropy(x_recon, x, size_average=False).div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld


use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('This code is running over', device)
max_iter = int(20)
batch_size = 100
z_dim = 500
lr_C = 0.001
beta1_C = 0.9
beta2_C = 0.999

training_set = datasets.CIFAR100(
    './tmp/CIFAR100', train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.CIFAR100(
    './tmp/CIFAR100', train=False, download=True, transform=transforms.ToTensor())

data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10000,
                         shuffle=True, num_workers=3)

VAE = CNNVAE1(z_dim).to(device)
VAE.load_state_dict(torch.load('./500_MMD_Plain_CNN_CIFAR10'))

C = Classification(z_dim).to(device)
optim_C = optim.Adam(C.parameters(), lr=0.001, betas=(beta1_C, beta2_C))

criterion = nn.CrossEntropyLoss()
print('Network is loaded')

A = Active_Units_Comp.Act_units(
    num_samples=20, sig=0.02).comp(net=VAE, input=test_loader)
print('number of active latent variables are: {:}'.format(A))

Result = []

for epoch in range(max_iter):
    train_loss = 0

    for batch_idx, (x_true, target) in enumerate(data_loader):

        x_true, target = x_true.to(device), target.to(device)
        z = VAE(x_true, no_dec=True)
        outputs = C(z)

        loss = criterion(outputs, target)

        optim_C.zero_grad()
        loss.backward()
        optim_C.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t  Loss: {:.6f}  '.format(epoch, batch_idx * len(x_true),
                                                                                len(data_loader.dataset),
                                                                                100. * batch_idx /
                                                                                len(data_loader),
                                                                                loss.item(),
                                                                                ))

    print('====> Epoch: {}, \t Average loss: {:.4f}'
          .format(epoch, train_loss / (batch_idx + 1)))

    [x_test, labels] = iter(test_loader).next()
    x_test, labels = x_test.to(device), labels.to(device)
    z = VAE(x_test.to(device), no_dec=True).squeeze()
    outputs = C(z)
    _, predicted = torch.max(outputs.data, 1)
    Accuracy = (predicted == labels).sum().item() / x_test.size(0)
    print('======> Epoch: {}, Accuracy: {:.6f}'.format(epoch, Accuracy))

    Result.append(('====>epoch:', epoch,
                   'loss:', train_loss / (batch_idx + 1),
                   'Test Accuracy:', Accuracy))

if z_dim == 2:
    batch_size_test = 500
    z_list, label_list = [], []

    for i in range(5):
        x_test, y_test = iter(test_loader).next()
        x_test = Variable(x_test, requires_grad=False).to(device)
        _, _, _, z = VAE(x_test)
        z_list.append(z.cpu().data.numpy())
        label_list.append(y_test.numpy())
    z = np.concatenate(z_list, axis=0)
    label = np.concatenate(label_list)

    frame1 = sns.kdeplot(z[:, 0], z[:, 1], n_levels=300, cmap='hot')
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('Info_Latent_Distirbutions.eps', format='eps', dpi=1000)
    plt.show()

    frame2 = plt.scatter(z[:, 0], z[:, 1], c=label,
                         cmap='jet', edgecolors='black')
    frame2.axes.get_xaxis().set_visible(False)
    frame2.axes.get_yaxis().set_visible(False)
    plt.colorbar()
    plt.savefig('Info_Latent_Distirbutions_Labels.eps', format='eps', dpi=1000)
    plt.show()

with open("Info_CIFAR10Accuracy.txt", "w") as output:
    output.write(str(Result))


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
