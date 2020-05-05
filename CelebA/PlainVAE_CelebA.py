import torch, os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch.nn.init as init
import math

class CNNVAE1(nn.Module):

    def __init__(self, z_dim=100):
        super(CNNVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 2*z_dim, 2, 1),

        )
        self.decode = nn.Sequential(
            nn.Conv2d(z_dim, 256, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):

        initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        if no_enc:
            gen_z = Variable(torch.randn(4, z_dim, 1, 1), requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

        else:
            stats = self.encode(x)
            mu = stats[:, :self.z_dim]
            logvar = stats[:, self.z_dim:]
            z = self.reparametrize(mu, logvar)
            x_recon = self.decode(z)
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
    samples = np.reshape(samples, [height, cnt, cnt, width, 3])
    samples = np.transpose(samples, axes=[1, 0, 2, 3, 4])
    samples = np.reshape(samples, [height*cnt, width*cnt, 3])
    return samples

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('This code is ruuning over', device)

image_size = 64
max_iter = int(20)
batch_size = 50
z_dim = 200
lr = 0.001
beta1 = 0.9
beta2 = 0.999

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = ImageFolder('./images/', transform)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True)

VAE = CNNVAE1().to(device)
optim = optim.Adam(VAE.parameters(), lr=lr, betas=(beta1, beta2))

Result = []

for epoch in range(max_iter):
    train_loss = 0
    KL_loss = 0
    Loglikehood_loss= 0

    for batch_idx, (x_true, y_true) in enumerate(data_loader):

        x_true = x_true.to(device)
        x_recon, mu, logvar, z = VAE(x_true)
        vae_recon_loss = recon_loss(x_recon, x_true)
        KL = kl_divergence(mu, logvar)

        loss = vae_recon_loss + KL

        train_loss += loss.item()
        Loglikehood_loss += vae_recon_loss.item()
        KL_loss += KL.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \t  Loss: {:.6f} \t Cross Entropy: {:.6f} \t KL Loss: {:.6f}'.format(epoch, batch_idx * len(x_true),
                                                                              len(data_loader.dataset),
                                                                              100. * batch_idx / len(data_loader),
                                                                              loss.item(),
                                                                              vae_recon_loss,
                                                                              KL))

        if batch_idx % 10 == 0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / (batch_idx + 1)))
            Result.append(('====>epoch:', epoch,
                               'loss:', train_loss / (batch_idx + 1),
                               'Loglikeihood:', Loglikehood_loss / (batch_idx + 1),
                               'KL:', KL_loss / (batch_idx + 1),
                               ))

with open("PlainResults.txt", "w") as output:
    output.write(str(Result))

torch.save(VAE.state_dict(), './New_Plain_VAE_CNN_CelebA')
print('The net\'s parameters are saved')
