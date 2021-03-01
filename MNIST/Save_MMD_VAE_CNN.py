

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


class CNNVAE1(nn.Module):

    def __init__(self, z_dim=2):
        super(CNNVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(28, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(28, 56, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(56, 118, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(118, 2 * z_dim, 1),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 118, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(118, 118, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(118, 56, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(56, 28, 4, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(28, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(28, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        if no_enc:
            gen_z = Variable(torch.randn(100, z_dim, 1, 1), requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

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


def convert_to_display(samples):
    cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('This code is ruuning over', device)

max_iter = int(10)
batch_size = 100
z_dim = 20
lr = 0.001
beta1 = 0.9
beta2 = 0.999
gamma = 1

training_set = datasets.MNIST('./tmp/MNIST', train=True, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

test_set = datasets.MNIST('./tmp/MNIST', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_set, batch_size=500, shuffle=True)

VAE = CNNVAE1(z_dim).to(device)
optim = optim.Adam(VAE.parameters(), lr=lr, betas=(beta1, beta2))

Result = []

for epoch in range(max_iter):

    train_loss = 0
    KL_loss = 0
    Loglikehood_loss= 0

    for batch_idx, (x_true, _) in enumerate(data_loader):
        x_true = x_true.to(device)
        x_recon, mu, logvar, z = VAE(x_true)

        P_z = Variable(torch.randn(batch_size, z_dim), requires_grad=False).to(device)
        mmd = compute_mmd(P_z, z)

        vae_recon_loss = recon_loss(x_recon, x_true)
        KL = kl_divergence(mu, logvar)
        loss = vae_recon_loss + 10000*mmd

        train_loss += loss.item()
        Loglikehood_loss += vae_recon_loss.item()
        KL_loss += KL.item()


        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t  Loss: {:.6f} \t Cross Entropy: {:.6f} \t KL Loss: {:.6f}'.format(epoch, batch_idx * len(x_true),
                                                                              len(data_loader.dataset),
                                                                              100. * batch_idx / len(data_loader),
                                                                              loss.item(),
                                                                              vae_recon_loss,
                                                                              KL))

    print('====> Epoch: {}, \t Average loss: {:.4f}, \t Log Likeihood: {:.4f}, \t KL: {:.4f} '
          .format(epoch, train_loss / (batch_idx + 1), Loglikehood_loss/ (batch_idx + 1), KL_loss/ (batch_idx + 1)))


    Result.append(('====>epoch:', epoch,
                   'loss:', train_loss / (batch_idx + 1),
                   'Loglikeihood:', Loglikehood_loss / (batch_idx + 1),
                   'KL:', KL_loss / (batch_idx + 1),
                   ))

#with open("file.txt", "w") as output:
#    output.write(str(Result))


torch.save(VAE.state_dict(), './Beta_VAE_CNN')
print('The net\'s parameters are saved')
