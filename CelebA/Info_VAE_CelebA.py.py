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
import warnings

warnings.filterwarnings('ignore')

class Discriminator(nn.Module):
    def __init__(self, z_dim=100):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(64*64*3 + z_dim, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 10),
            nn.LeakyReLU(0.2, True),
            nn.Linear(10, 1),

        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x, z):
        x = x.view(-1, 64*64*3)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()



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

def permute_dims(z):
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B).to(device)
    perm_z = z[perm]
    return perm_z

def convert_to_display(samples):
    cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width, 3])
    samples = np.transpose(samples, axes=[1, 0, 2, 3, 4])
    samples = np.reshape(samples, [height*cnt, width*cnt, 3])
    return samples

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('This code is running over', device)

image_size = 64
max_iter = int(10)
batch_size = 50
z_dim = 100
lr_D = 0.001
beta1_D = 0.9
beta2_D = 0.999
gamma = 10

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = ImageFolder('./images/', transform)
data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True)

VAE = CNNVAE1().to(device)
D = Discriminator().to(device)

optim_VAE = optim.Adam(VAE.parameters(), lr=lr_D, betas=(beta1_D, beta2_D))
optim_D = optim.Adam(D.parameters(), lr=0.0001, betas=(beta1_D, beta2_D))

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

with open("20InfoResults.txt", "w") as output:
    output.write(str(Result))


torch.save(VAE.state_dict(), './New_Info_VAE_CNN_CelebA')
print('The net\'s parameters are saved')