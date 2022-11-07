import torch.nn as nn
import torch
from torch.autograd import Variable

class discriminator(nn.Module):
    
    """discriminator network.
    Args:
        z_dim (int): dimension of latent code (typically a number in [10 - 256])
        x_dim (int): for example m x n x c for [m, n, c]
    """
    def __init__(self, z_dim=2, x_dim=784):
        super(discriminator, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.net = nn.Sequential(
            nn.Linear(self.x_dim + self.z_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 400),
            nn.ReLU(True),
            nn.Linear(400, 100),
            nn.ReLU(True),
            nn.Linear(100, 1),
        )

    def forward(self, x, z):
        """
        Inputs:
            x : input from train_loader (batch_size x input_size )
            z : latent codes associated with x (batch_size x z_dim)
        """
        x = x.view(-1, self.x_dim)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()


class cnn_mnist(nn.Module):
    
    """CNN for MNIST.
    Args:
        z_dim (int): dimension of latent codes
    """

    def __init__(self, x_dim=784, z_dim=2):
        super(cnn_mnist, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.encode = nn.Sequential(
            nn.Conv2d(1, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(28, 28, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(28, 56, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(56, 118, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(118, 2 * self.z_dim, 1),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(self.z_dim, 118, 1),
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
        """
        inputs:
            mu: mean of q(z|x) (encoder)
            logvar: log of vairance of q(z|x)
        
        output:
            z: samples from q(z|x)
        """
        
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        
        """
        inputs:
            x (float matrix): inputs
            no_enc (bool): for generation purpose without having encoding engaged  
        """
        if no_enc:
            gen_z = Variable(torch.randn(100, z_dim, 1, 1),
                             requires_grad=False)
            gen_z = gen_z.to(device)
            return self.decode(gen_z).view(x.size())

        else:
            stats = self.encode(x)
            mu = stats[:, :self.z_dim]
            logvar = stats[:, self.z_dim:]
            z = self.reparametrize(mu, logvar)
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()


class mlp_mnist(nn.Module):

    def __init__(self, x_dim=784, z_dim=2):
        super(mlp_mnist, self).__init__()
        """
        Args:
            x_dim (int): dimension of flatten input (for example for X[m, n, c] is m x n x c)
            z_dim (int): dimension of latent codes
        """
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.encode = nn.Sequential(
            nn.Linear(self.x_dim, 1000),
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
            nn.Linear(1000, 2 * self.z_dim),
        )
        self.decode = nn.Sequential(
            nn.Linear(self.z_dim, 1000),
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
            nn.Linear(2000, self.x_dim),
            nn.Sigmoid(),
        )

    def reparametrize(self, mu, logvar):
        """
        inputs:
            x (float matrix): inputs
            no_enc (bool): for generation purpose without having encoding engaged  
        """
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_enc=False):
        """
        inputs:
            x (float matrix): inputs
            no_enc (bool): for generation purpose without having encoding engaged
        """
        if no_enc:
            gen_z = Variable(torch.randn(100, self.z_dim), requires_grad=False)
            gen_z = gen_z.to(x.device())
            return self.decode(gen_z).view(x.size())

        else:
            stats = self.encode(x.view(-1, self.x_dim))
            mu = stats[:, :self.z_dim]
            logvar = stats[:, self.z_dim:]
            z = self.reparametrize(mu, logvar)
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()
        

class cnn_cifar(nn.Module):
    """
    Args:
        z_dim (int): dimension of latent codes
        n_channel(int): input channels
    """
    def __init__(self, z_dim=2, n_channels=3):
        super(cnn_cifar, self).__init__()
        self.z_dim = z_dim
        self.n_channels = n_channels
        self.encode = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 2 * self.z_dim, 1),
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
            nn.ConvTranspose2d(32, self.n_channels, 4, 2, 1),
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
