import torch
from utils.utils import AverageMeter, ProgressMeter, permute_dims, compute_mmd
import time
from torch.autograd import Variable
from utils import loss




def train_infomax(train_loader, vae, disc, optim_vae, optim_disc, epoch, args):
    """
        function: one epoch training of infomax model 
          
        Args:
        train_loader (class): train loader
        vae (torch nn.module): variational autoencoder models
        disc (torch nn.module): discriminator to calculating mutual information
        optim_vae, opim_disc (): optimizer
        epoch (int): training epochs
        args: training arguments
        
        for more information check this paper:
        Rezaabad, Ali Lotfi, and Sriram Vishwanath. "Learning representations by maximizing mutual information in variational autoencoders." 2020 IEEE International Symposium on Information Theory (ISIT). IEEE, 2020.    
    """
    #  tracking time of loading and losses
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('losses', ':6.2f')
    info_loss = AverageMeter('info_loss', ':6.2f')
    vae_loss = AverageMeter('vae_loss', ':6.2f')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, info_loss, vae_loss],
                             prefix="Epoch: [{}]".format(epoch))

    # activating training
    vae.train()
    disc.train()
    end = time.time()

    for batch_idx, (x_true, _) in enumerate(train_loader):

        # measuring data loading time for
        data_time.update(time.time() - end)

        # assigning device to Inputs
        x_true = x_true.to(args.device)

        # pass samples x_true from the vae
        x_recon, mu, logvar, z = vae(x_true)

        # pass x_true and learned features z from the discriminator
        d_xz = disc(x_true, z)

        z_perm = permute_dims(z)
        d_x_z = disc(x_true, z_perm)

        info_xz = -(d_xz.mean() - (torch.exp(d_x_z - 1).mean()))

        vae_recon_loss = loss.recon_loss(x_recon, x_true)
        vae_kld = loss.kl_divergence(mu, logvar)

        vae_loss_ = vae_recon_loss + args.alpha * vae_kld
        infomax_loss = vae_loss_ + args.gamma * info_xz

        optim_vae.zero_grad()
        infomax_loss.backward(retain_graph=True)
        optim_vae.step()

        optim_disc.zero_grad()
        info_xz.backward(inputs=list(disc.parameters()))
        optim_disc.step()
        
        # update measurements to display the results

        batch_time.update(time.time() - end)
        losses.update(infomax_loss, args.batch_size)
        info_loss.update(info_xz, args.batch_size)
        vae_loss.update(vae_loss_, args.batch_size)
        
        
        if (batch_idx % args.print_freq == 0) or (batch_idx == len(train_loader) - 1):
            progress.display(batch_idx)
            
            
def train_vae(train_loader, vae, optim_vae, epoch, args):
    """
        function: one epoch training of vae
          
        Args:
        train_loader (class): train loader
        vae (torch nn.module): variational autoencoder models
        optim_vae : optimizer
        epoch (int): training epochs
        args: training arguments
    
        for more information check this paper:
        Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
    """
    #  tracking time of loading and losses
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('losses', ':6.2f')
    kl_loss = AverageMeter('kl_loss', ':6.2f')
    recon_loss = AverageMeter('vae_recon_loss', ':6.2f')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, kl_loss, recon_loss],
                             prefix="Epoch: [{}]".format(epoch))

    # activating training
    vae.train()
    end = time.time()

    for batch_idx, (x_true, _) in enumerate(train_loader):

        # measuring data loading time for
        data_time.update(time.time() - end)

        # assigning device to Inputs
        x_true = x_true.to(args.device)

        # pass samples x_true from the vae
        x_recon, mu, logvar, z = vae(x_true)


        recon_loss_ = loss.recon_loss(x_recon, x_true)
        kld = loss.kl_divergence(mu, logvar)

        vae_loss_ = recon_loss_ + args.alpha * kld

        optim_vae.zero_grad()
        vae_loss_.backward()
        optim_vae.step()


        # update measurements to display the results

        batch_time.update(time.time() - end)
        losses.update(vae_loss_, args.batch_size)
        kl_loss.update(kld, args.batch_size)
        recon_loss.update(recon_loss_, args.batch_size)

        if (batch_idx % args.print_freq == 0) or (batch_idx == len(train_loader) - 1):
            progress.display(batch_idx)


def train_mmd(train_loader, vae, optim_vae, epoch, args):
    """
        function: one epoch training of vae
          
        Args:
        train_loader (class): train loader
        vae (torch nn.module): variational autoencoder models
        optim_vae : optimizer
        epoch (int): training epochs
        args: training arguments
    
        for more information check this paper:
        Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
    """
    #  tracking time of loading and losses
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('losses', ':6.2f')
    mmd_loss = AverageMeter('mmd_loss', ':6.2f')
    recon_loss = AverageMeter('recon_loss', ':6.2f')

    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, mmd_loss, recon_loss],
                             prefix="Epoch: [{}]".format(epoch))

    # activating training
    vae.train()
    end = time.time()

    for batch_idx, (x_true, _) in enumerate(train_loader):

        # measuring data loading time for
        data_time.update(time.time() - end)

        # assigning device to Inputs
        x_true = x_true.to(args.device)

        # pass samples x_true from the vae
        x_recon, mu, logvar, z = vae(x_true)
        p_z = Variable(torch.randn(args.batch_size, args.dim), requires_grad=False).to(args.device)
        
        mmd = compute_mmd(p_z, z)
        recon_loss_ = loss.recon_loss(x_recon, x_true)

        vae_loss_ = recon_loss_ + args.alpha * mmd

        optim_vae.zero_grad()
        vae_loss_.backward()
        optim_vae.step()

        # update measurements to display the results

        batch_time.update(time.time() - end)
        losses.update(vae_loss_, args.batch_size)
        mmd_loss.update(mmd, args.batch_size)
        recon_loss.update(recon_loss_, args.batch_size)

        if (batch_idx % args.print_freq == 0) or (batch_idx == len(train_loader) - 1):
            progress.display(batch_idx)
