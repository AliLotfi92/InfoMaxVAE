
import torch.nn.functional as F


def recon_loss(x_recon, x):
    """
    inputs:
        x_recon: recontruced image (output of the decoder)
        x      : input   
    """
    batch_size = x.size(0)
    loss = F.binary_cross_entropy(x_recon, x, size_average=False).div(batch_size)
    return loss


def kl_divergence(mu, logvar):
    """
    inputs:
        mu    : mean of q(z|x)
        logvar: logvar of q(z|x)
    """
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld
