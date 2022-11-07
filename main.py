import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import warnings
from models import model_vae
from utils import utils
from utils.utils import save_checkpoint
import argparse
from dataset import dataset_
from utils import train


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default='mmd', choices=['vae', 'infomax', 'mmd'], help='choose a method among [mmd, infomax, vae]')
parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'], help='type of encoder and decoder')
parser.add_argument('--seed', type=int, default=123456789, help='Random seed')
parser.add_argument('--dim', type=int, default=10, help='dim for latent codes [typical values: 10 for mnist, 28 for cifar]')
parser.add_argument('--num-iters', type=int, default=100, help='number of epochs')
parser.add_argument('--batch-size', type=int, default=64, help='size of batch')
parser.add_argument('--batch-size-test', type=int, default=500, help='size of batch for validation')
parser.add_argument('--gamma', type=float, default=20., help='information preference coefficient')
parser.add_argument('--alpha', type=float, default=1.,help='KL divergence coefficient')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the vae (encoder and decoder)')
parser.add_argument('--print-freq', type=int, default=100, help='frequency to show the progress based on batch index')
parser.add_argument('--beta1', type=float, default=0.9, help='beta 1 for adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta 2 for adam optimizer')
parser.add_argument('--lr-dis', type=float, default=0.00001, help='learning for mutual information estimator network')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'], help='name of dataset')
parser.add_argument('--data-dir', type=str, default='./data', help='directory of dataset (or to save)')
parser.add_argument('--save', default='save', type=str, help='save root')
parser.add_argument('--save-freq', type=int, default=5, help='frequency to save the progress based on batch index')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--trial', default=1, type=int, help='auxiliary string to distinguish trials')

def main(args):
    
    print(args)
    
    # select device; cuda is gpu is available
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    args.device = device
    print(f'method: {args.method}')
    print('device: {}'.format(device))
    
    # load dataset
    training_set, test_set, classes, num_classes = dataset_.loader(args)
    print(f'dataset loaded: {args.dataset} \n number of classes: {classes} \n classes: {num_classes}')
    
    
    # data loders 
    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_test, shuffle=False, num_workers=3)

    # select the model based on the arguments
    if args.dataset == 'mnist' and args.model == 'mlp':
        vae = model_vae.mlp_mnist(x_dim=args.x_dim, z_dim=args.dim)  
    elif args.dataset == 'mnist' and args.model == 'cnn':
        vae = model_vae.cnn_mnist(x_dim=args.x_dim, z_dim=args.dim)
    elif args.dataset in ['cifar10', 'cifar100']:
        vae = model_vae.cnn_cifar(x_dim=args.x_dim, z_dim=args.dim)
    
    # calculating the number of trainable parameters of vae and discriminator
    total_trainable_param = utils.parameters(vae)
    print('total number of trainable parameres in model: {}'.format(total_trainable_param))
    
    if args.method == 'infomax':  
        # initialize a discriminator for infomax vae
        disc = model_vae.discriminator(z_dim=args.dim, x_dim=args.x_dim)
        total_param_disc = utils.parameters(disc)
        print('total number of parameters in discriminator: {}'.format(total_param_disc))
        
        # assign selected device to discriminator
        disc = disc.to(args.device)
        
        # Adam optimzer for discriminator
        optim_disc = optim.Adam(disc.parameters(), lr=args.lr_dis, betas=(args.beta1, args.beta2))
        
        
    # assign selected device to vae
    vae = vae.to(args.device)
    
    # initializing the parameters of adam optimizer for vae
    optim_vae = optim.Adam(vae.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    # make a name for directory to save training file
    args.model_name = f'{args.save}/{args.dataset}/method_{args.method}_model_{args.model}_latent_dim_{args.dim}_lr_vae_{args.lr}_disc_lr_{args.lr_dis}_gamma_{args.gamma}_alpha_{args.alpha}_iterations_{args.num_iters}_trial_{args.trial}'
    
        
    if not os.path.exists(args.model_name):
        os.makedirs(args.model_name)
    
    
    for epoch in range(args.start_epoch, args.num_iters):
        
        # select the model and save its associated parameters for each iteration (replacing)
        
        if args.method == 'infomax':
            train.train_infomax(train_loader, vae, disc, optim_vae, optim_disc, epoch, args)
            if epoch % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model,
                    'vae_state_dict': vae.state_dict(),
                    'disc_state_dict': disc.state_dict(),
                    'optim_vae': optim_vae.state_dict(),
                    'optim_disc': optim_disc.state_dict(),
                    }, filename=f'{args.model_name}/checkpoint.pth')
                
        elif args.method == 'vae':
            train.train_vae(train_loader, vae, optim_vae, epoch, args)
            if epoch % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model,
                    'vae_state_dict': vae.state_dict(),
                    'optim_vae': optim_vae.state_dict(),
                    }, filename=f'{args.model_name}/checkpoint.pth')
        
        elif args.method == 'mmd':
            train.train_mmd(train_loader, vae, optim_vae, epoch, args)
            if epoch % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.model,
                    'vae_state_dict': vae.state_dict(),
                    'optim_vae': optim_vae.state_dict(),
                    }, filename=f'{args.model_name}/checkpoint.pth')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

