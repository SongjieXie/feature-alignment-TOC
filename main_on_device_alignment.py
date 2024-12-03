import os
import itertools
import torch
import torch.nn as nn 
from tensorboardX import SummaryWriter
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler

from dataloader import get_data

from utils import get_models_with_on_device_alignment, load_anchors
from models.channel_model import AWGN_complex

from train import Trainer_relative


def main(args):
    """ Model and Opt """
    model_enc, model_dec = get_models_with_on_device_alignment(args.model,
                                                               args.sc,
                                                               args.ratio)
        
    channel_m = AWGN_complex()
    
    models_params = itertools.chain(model_enc.parameters(), model_dec.parameters())
    optimizer = optim.AdamW(params=models_params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.5) 
    
    
    """ Criterion """
    criterion = nn.CrossEntropyLoss()
    
    """ dataloader """
    dataloader_train =  get_data(args.dataset, args.N,
                                 n_worker= args.num_workers)
    dataloader_vali = get_data(args.dataset, 1000, 
                               n_worker= args.num_workers, train=False)
    anchors = load_anchors(args) 
    
    """ writer """
    root_d = '{0}/log/'.format(args.root) if args.train == 1 else '{0}/results/'.format(args.root)
    log_writer = SummaryWriter(root_d + name)
    
    """ Training """
    snrs = [args.snr, args.snr]
    trainer = Trainer_relative(args.dataset, model_enc, model_dec, anchors, channel_m,
                      dataloader_train, dataloader_vali, optimizer, scheduler, criterion, 
                      args.device, args.maxnorm)
    trainer.train(args.epoches, log_writer, snrs, path=path_to_backup, is_vali=True, is_display=True)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    

    parser = argparse.ArgumentParser(description='Hypernet-DJSCC')
    parser.add_argument('--id', type=int, default=3, help='ID')
    parser.add_argument('--model', type=str, default='vgg', help='model: vgg or resnet')
    
    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('--sc', type=int, default=16, help='The number of channels for symbol transmission, or the number of symbols for taks-oriented')
    parser.add_argument('--channel', type=str, default='awgn', help='The channel models including awgn and fading')
    
    # only for the base model
    parser.add_argument('--snr', type=float, default=10.0, help='The snr of AWGN channel when training base model')
    
    parser.add_argument('-r', '--root', type=str, default='./local_relative', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cuda:0', help= 'The device')
    
    parser.add_argument('--train', type=int, default=1, help='1: train, 0: evaluation')
    parser.add_argument('-e', '--epoches', type=int, default=200, help='Number of epoches')
    parser.add_argument('--N', type=int, default=256, help='The batch size of training data')
    parser.add_argument('--lr', type=float, default=1e-3, help='learn rate')
    parser.add_argument('--maxnorm', type=float, default=1., help='The max norm of flip')
    parser.add_argument('--step', type=int, default=80, help='learn rate')
    parser.add_argument('--ratio', type=int, default=2, help='')
    
    

    parser.add_argument('--num_workers', type=int, default=int(mp.cpu_count()/4),
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    
    args = parser.parse_args()
    args.n_iter = 0
    
    print('>>> The number of workers (default: {0})'.format(args.num_workers))
    
    name = args.model + '-' +str(args.id) + '-' + args.channel + '-' + str(args.snr) +'-'+ str(args.sc) + '-' + str(args.ratio)+ '-' + str(args.dataset)
    print(">>> The name of the trained model: ", name)
 
    path_to_backup = os.path.join(args.root, name)
    if not os.path.exists(path_to_backup):
        print('>>> Making ', path_to_backup, '...')
        os.makedirs(path_to_backup)

    args.device = torch.device(args.device if(torch.cuda.is_available()) else "cpu")
    print('>>> Device: ', args.device)
    
    args.anchors_path = os.path.join(args.root, args.dataset)
    
    main(args)