import os
import argparse
import itertools
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
import time 



import multiprocessing as mp

from train import get_repr, Aligner

from dataloader import get_data

# from utils import get_models_2
from models.channel_model import AWGN_complex
from models.align_models import AlignerLinear

from utils import get_models, get_path_2, load_enc, load_dec, evaluate, evaluate_with_alignment


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default='./local', help='The root of trained models')
    parser.add_argument('--enc_path', type=str, default=1, help='The path for the on-device encoder')
    parser.add_argument('--dec_path', type=str, default=1, help='The path for the server-based inference network')
    
    parser.add_argument('-d', '--dataset', type=str, default='SVHN', help='dataset name')
    
    parser.add_argument('--device', type=str, default='cuda:0', help= 'The device')
    
    parser.add_argument('--align_method', type=str, default='ls', help='The method for server-based alignment')
    parser.add_argument('--N', type=int, default=100, help='The batch size of training data')
    parser.add_argument('--num_workers', type=int, default=int(mp.cpu_count()/1.5),
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    return parser.parse_args()


def my_align(enc, enc_2, dec, dataloader_vali, channel_m, snrs, sc, config):
    dataloader_tmp = get_data(config.dataset, config.N, 
                               n_worker= config.num_workers, 
                               train=True)
    anchors_x, anchors_y = next(iter(dataloader_tmp))
    
    anchors_x, anchors_y = anchors_x.to(config.device), anchors_y.to(config.device)
    
    dataloader_vali = get_data(config.dataset, 1000, 
                               n_worker= config.num_workers, 
                               train=False)
    
    z_1, z_2, tilde_z_1, tilde_z_2 = get_repr(enc, enc_2, anchors_x, anchors_y, channel_m, snrs)
    
    aligner = Aligner(config.align_method, sc, dec)
    
    aligner.fit(tilde_z_1, tilde_z_2, anchors_y, config)
    
    
    return evaluate_with_alignment(enc, dec, aligner, dataloader_vali, channel_m, 
             snrs, config.device) 


def main(config):
    """ Load models for encoder and decoder """
    info_list_enc = config.enc_path.split('-')
    info_list_dec = config.dec_path.split('-')
    
    assert info_list_enc[-1] == info_list_dec[-1], 'the dimension of transmitted feature vector should be equal'
    sc = int(info_list_enc[-1])
    enc, _ = get_models(info_list_enc[0], sc)
    enc_2, dec = get_models(info_list_dec[0], sc)
    
    name_enc = os.path.join(config.root, config.enc_path)
    name_dec = os.path.join(config.root, config.dec_path)
    
    enc, enc_2, dec = load_enc(enc, name_enc).to(config.device), load_enc(enc_2, name_dec).to(config.device), load_dec(dec, name_dec).to(config.device)
    
    """ Channel model """
    channel_m = AWGN_complex()
    # assert info_list_enc[3] == info_list_dec[3],
    snrs = [float(info_list_dec[3]), float(info_list_dec[3])]
    config.snr = float(info_list_dec[3])
    
    """ Dataset """
    dataloader_vali = get_data(config.dataset, 1000, 
                               n_worker= config.num_workers, 
                               train=False)
    
    if name_enc == name_dec:
        return evaluate(enc, dec, dataloader_vali, channel_m, snrs, config.device)
    
    return my_align(enc, enc_2, dec, dataloader_vali, channel_m, snrs, sc, config)
    
    
if __name__ == "__main__":
    config = parse_config()
    acc = main(config)
    print('Cross-model accuracy with server-based alignment: ', acc)
    
    
    
    
    
    
    
    
    
    
    
    