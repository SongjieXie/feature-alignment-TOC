import torch
import torchvision.datasets as dsets
import os

import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) 
        
def simple_transform_mnist():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

def simple_transform_cifar10(s): 
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])
def simple_transform_test_cifar10(s):
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
            ])
    
def simple_transform_SVHN(): 
    return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
def simple_transform_test_SVHN():
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])

 
root = r'/home/sxieat/data/'
def get_data(data_set, batch_size, shuffle=True, n_worker=0, train = True, add_noise=0):
    if data_set == 'CIFAR10':
        if train:
            tran = simple_transform_cifar10(32)
        else:
            tran = simple_transform_test_cifar10(32)
        t_tran =  None
        dataset = dsets.CIFAR10(root+'CIFAR10/', train=train, transform=tran, target_transform=t_tran, download=False) 
        
    elif data_set == 'SVHN':
        if train:
            tran = simple_transform_SVHN()
            split_str = 'train'
        else:
            tran = simple_transform_test_SVHN()
            split_str = 'test'
        t_tran =  None
        dataset = dsets.SVHN(root+'SVHN/', split=split_str, transform=tran, target_transform=t_tran, download=False) 
    else:
        print('Sorry! Cannot support ...')
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_worker)
    return dataloader
    


        
