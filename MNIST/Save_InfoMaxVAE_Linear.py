

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
import pickle

# Calling Seaborn causes pytorch warnings to be repeated in each loop, so I turned off these redudant warnings, but make sure
# you do not miss something important.

warnings.filterwarnings('ignore')


class Discriminator(nn.Module):
    def __init__(self, z_dim=2):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(784 + z_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 400),
            nn.ReLU(True),
            nn.Linear(400, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),

        )

    def forward(self, x, z):
        x = x.view(-1, 784)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()


class VAE1(nn.Module):

    def __init__(self, z_dim=2):
        super(VAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 2 * z_dim),
        )
        self.decode = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 2000),
            nn.ReLU(True),
            nn.Linear(2000, 784),
            nn.Sigmoid(),
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        if no_enc:
            gen_z = Variable(torch.randn(100, z_dim), requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

        else:
            stats = self.encode(x.view(-1, 784))
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


def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B).to(device)
    perm_z = z[perm]
    return perm_z


use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('This code is running over', device)
max_iter = int(5)
batch_size = 100
z_dim = 2
lr_D = 0.0001
beta1_D = 0.9
beta2_D = 0.999
gamma = 10


training_set = datasets.MNIST('./tmp/MNIST', train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('./tmp/MNIST', train=False, download=True, transform=transforms.ToTensor())

data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=500, shuffle=True, num_workers=3)

VAE = VAE1().to(device)
D = Discriminator().to(device)
optim_VAE = optim.Adam(VAE.parameters(), lr=lr_D, betas=(beta1_D, beta2_D))
optim_D = optim.Adam(D.parameters(), lr=0.00001, betas=(beta1_D, beta2_D))

Result = []

for epoch in range(max_iter):

    train_loss = 0
    train_info_loss = 0
    Loglikehood_loss = 0
    KL_loss = 0

    for batch_idx, (x_true,_) in enumerate(data_loader):

        x_true = x_true.to(device)
        x_recon, mu, logvar, z = VAE(x_true)

        D_xz = D(x_true, z)
        z_perm = permute_dims(z)
        D_x_z = D(x_true, z_perm)

        Info_xz = -(D_xz.mean() - (torch.exp(D_x_z - 1).mean()))

        vae_recon_loss = recon_loss(x_recon, x_true)
        vae_kld = kl_divergence(mu, logvar)

        vae_loss = vae_recon_loss + vae_kld + gamma * Info_xz

        optim_VAE.zero_grad()
        vae_loss.backward(retain_graph=True)
        optim_VAE.step()

        info_loss = Info_xz

        train_loss += vae_loss.item()
        Loglikehood_loss += vae_recon_loss.item()
        KL_loss += vae_kld.item()
        train_info_loss += -info_loss.item()

        optim_D.zero_grad()
        info_loss.backward()
        optim_D.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t  Loss: {:.6f} \t Cross Entropy: {:.6f} \t KL Loss: {:.6f} \t Info: {:.6f}'.format(epoch, batch_idx * len(x_true),
                                                                              len(data_loader.dataset),
                                                                              100. * batch_idx / len(data_loader),
                                                                              vae_loss.item(),
                                                                              vae_recon_loss.item(),
                                                                              vae_kld.item(),
                                                                              -Info_xz.item()))

    print('====> Epoch: {}, \t Average loss: {:.4f}, \t Log Likeihood: {:.4f}, \t KL: {:.4f} \t Info: {:.4f}'
          .format(epoch, train_loss / (batch_idx + 1), Loglikehood_loss/ (batch_idx + 1), KL_loss/ (batch_idx + 1), train_info_loss/(batch_idx + 1)))


    Result.append(('====>epoch:', epoch,
                   'loss:', train_loss / (batch_idx + 1),
                   'Loglikelihood:', Loglikehood_loss / (batch_idx + 1),
                   'KL:', KL_loss / (batch_idx + 1),
                   'Info:', train_info_loss/ (batch_idx + 1)
                   ))

with open("file.txt", "w") as output:
    output.write(str(Result))


torch.save(VAE.state_dict(), './Info_VAE_Linear')
print('The net\'s parameters are saved')
