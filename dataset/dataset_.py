
from torchvision import datasets, transforms
import os


def loader(args):
    """
        This function returns train/test sets, classes, num_classes
        input: args
        output: train_set, test_set, c
    """
    transform = transforms.ToTensor()
    data_dir = os.path.join(args.data_dir, args.dataset)
    
    if args.dataset == 'mnist':
        training_set = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform)
        args.x_dim = 28*28
        
    elif args.dataset == 'cifar10':
        training_set = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform)
        args.x_dim = 3*32*32
        
    elif args.dataset == 'cifar100':
        training_set = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=transform)
        args.x_dim = 3*32*32    
    else:
        assert 'Not implemented'
        
    class_info = training_set.classes
    class_num = len(class_info)
        
    return training_set, test_set, class_info, class_num