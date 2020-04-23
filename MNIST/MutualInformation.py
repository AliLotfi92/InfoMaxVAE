

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
        self.weight_init()

    def weight_init(self, mode='normal'):
        initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x, z):
        x = x.view(-1, 784)
        z = z.view(-1, z_dim)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()



class VAE1(nn.Module):

    def __init__(self, z_dim=2):
        super(VAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Linear(784, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2 * z_dim),
        )
        self.decode = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 784),
            nn.Sigmoid(),
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        if no_enc:
            gen_z = Variable(torch.randn(49, z_dim), requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

        else:
            stats = self.encode(x.view(-1, 784))
            mu = stats[:, :self.z_dim]
            logvar = stats[:, self.z_dim:]
            z = self.reparametrize(mu, logvar)
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

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

def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B).to(device)
    perm_z = z[perm]
    return perm_z


use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('This code is running over', device)
max_iter = int(1)
batch_size = 100
z_dim = 2
lr_D = 0.001
beta1_D = 0.9
beta2_D = 0.999
gamma = 10



training_set = datasets.MNIST('./tmp/MNIST', train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('./tmp/MNIST', train=False, download=True, transform=transforms.ToTensor())

data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10000, shuffle=True, num_workers=3)


VAE = VAE1().to(device)
VAE.load_state_dict(torch.load('./Info_VAE_Linear'))

D = Discriminator().to(device)
optim_D = optim.Adam(D.parameters(), lr=0.00001, betas=(beta1_D, beta2_D))


print('Network is loaded')

print('Network is loaded')
A = Active_Units_Comp.Act_units(num_samples=20, sig=0.02).comp(net=VAE, input=test_loader)
print('number of active latent variables are: {:}'.format(A))
Result = []


for epoch in range(max_iter):
    train_info_loss = 0

    for batch_idx, (x_true,_) in enumerate(data_loader):

        x_true = x_true.to(device)
        _, _, _, z = VAE(x_true)

        D_xz = D(x_true, z)
        z_perm = permute_dims(z)
        D_x_z = D(x_true, z_perm)

        Info_xz = -(D_xz.mean() - (torch.exp(D_x_z - 1).mean()))


        info_loss = Info_xz
        train_info_loss += -info_loss.item()

        optim_D.zero_grad()
        info_loss.backward()
        optim_D.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t  Loss: {:.6f}  \t Info: {:.6f}'.format(epoch, batch_idx * len(x_true),
                                                                              len(data_loader.dataset),
                                                                              100. * batch_idx / len(data_loader),
                                                                              Info_xz.item(),
                                                                              -Info_xz.item(),))

    print('====> Epoch: {}, \t Average loss: {:.4f},  \t Info: {:.4f}'
          .format(epoch, -train_info_loss / (batch_idx + 1), train_info_loss / (batch_idx + 1)))


    Result.append(('====>epoch:', epoch,
                   'loss:', -train_info_loss / (batch_idx + 1),
                   'Info:', train_info_loss / (batch_idx + 1)
                   ))

(x_test, _ ) = iter(test_loader).next()
x_test = x_test.to(device)
_, mu, logvar, z = VAE(x_test.to(device))

vae_kld = kl_divergence(mu, logvar)
print(vae_kld)

D_xz = D(x_test, z)
z_perm = permute_dims(z)
D_x_z = D(x_test, z_perm)

Info_xz = -(D_xz.mean() - (torch.exp(D_x_z - 1).mean()))
print(Info_xz.item())


x_recon, _, _, _ = VAE(x_true)
recon_error = recon_loss(x_recon, x_true)
print(recon_error)

Result.append(('====>epoch:', epoch,
               'loss:', Info_xz.item() / (batch_idx + 1),
               'Info:', -Info_xz.item() / (batch_idx + 1),
               'RE:', recon_error .item()))

with open("InfoLinear.txt", "w") as output:
    output.write(str(Result))
