import time
import torch
import torch.nn as nn 
import torch.optim as optim 
import os
from utils import accuracy, psnr, load_model, save_model, sample
from models.align_models import AlignerLinear, AlignerNonLinear

class Trainer():
    
    def __init__(self, dataname, model_enc, model_dec, channel,
                 datal_train, datal_vali, optimizer,scheduler, criterion,
                 device, maxnorm):
        self.dataname = dataname
        
        self.datal_train = datal_train
        self.datal_vali = datal_vali
        self.model_enc = model_enc
        self.model_dec = model_dec
        
        self.channel = channel
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        self.device = device
        self.maxnorm = maxnorm
        
        
    def train(self, epoches, writer, snrs,
              current_epoch=0, save_model=True, path=None, 
              is_vali=True, is_display=True, load_model=False):
        if load_model:
            current_epoch = self.__load(path)
        
        n_iter =  0
        best_acc = 0.0
        max_iter = len(self.datal_train) * epoches
        N = len(self.datal_train.dataset)
        for epoch in range(current_epoch, epoches):
            self.model_enc.to(self.device)
            self.model_enc.train()
            self.model_dec.to(self.device)
            self.model_dec.train()
            self.channel.train()
            self.criterion.train()
            
            epoch_start = time.time()
            # Training
            re = self.train_one_epoch(n_iter, writer, snrs, max_iter, N)
            acc = self.validate(snrs)
            
            self.scheduler.step()
            epoch_end = time.time()
            time_for_one_epoch = epoch_end - epoch_start
            
            if save_model and ((epoch == 0) or (acc > best_acc)):
                best_acc = acc
                self.__save(epoch, path)
                
            if is_display:
                print('\n \n Time for one epoch',':', time_for_one_epoch)
                batches_start = time.time()
                print('[%d:%d]\t Losses:%.4f\t'% (epoch, len(self.datal_train), re.item()))
                print('ACC>>>>>:', acc)
   
        
    def train_one_epoch(self, n_iter, writer, snrs:list, max_iter, N):
        for i_batch, (x, labs) in enumerate(self.datal_train):
            x = x.to(self.device)
            labs = labs.to(self.device)
            num_imgs = x.shape[0]
            snrs = sample(snrs, num_imgs)
                
            z = self.model_enc(x)
            z = self.channel(z, snrs)
            out = self.model_dec(z)
            loss = self.__compute_loss(out, x,labs)
                
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.maxnorm > 0:
                torch.nn.utils.clip_grad_norm_(self.model_enc.parameters(), self.maxnorm)
                torch.nn.utils.clip_grad_norm_(self.model_dec.parameters(), self.maxnorm)
            self.optimizer.step()
            
            writer.add_scalar('train/losses', loss.item(), n_iter)
            n_iter += 1
            
        return loss
            
    
    def validate(self, snrs):
        acc = 0
        with torch.no_grad():
            self.model_enc.eval()
            self.model_dec.eval()

            for x, labs in self.datal_vali:
                x = x.to(self.device) 
                labs = labs.to(self.device)
                
                num_imgs = x.shape[0]
                snrs = sample(snrs, num_imgs)
                
                z_mean = self.model_enc(x)
                z = self.channel(z_mean, snrs)
                out = self.model_dec(z)
                acc += accuracy(out, labs)[0].item()
                
        return acc / len(self.datal_vali)
                
                
    def __compute_loss(self, out, x, labs):
        return self.criterion(out, labs)
                
    def __load(self, path=None):
        return load_model(self.model_enc, model_dec, self.optimizer, path)
       
    def __save(self, epoch, path=None):
        save_model(epoch, self.model_enc, self.model_dec, self.optimizer, path=path)  
        
        
        
class Trainer_relative():
    
    def __init__(self, dataname, model_enc, model_dec, anchors, channel,
                 datal_train, datal_vali, optimizer,scheduler, criterion,
                 device, maxnorm):
        self.dataname = dataname
        
        self.datal_train = datal_train
        self.datal_vali = datal_vali
        self.model_enc = model_enc
        self.model_dec = model_dec
        
        self.anchors = anchors.to(device)
        
        self.channel = channel
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
        self.device = device
        self.maxnorm = maxnorm
        

    def train(self, epoches, writer, snrs,
              current_epoch=0, save_model=True, path=None, 
              is_vali=True, is_display=True, load_model=False):
        if load_model:
            current_epoch = self.__load(path)
        
        n_iter =  0
        best_acc = 0.0
        max_iter = len(self.datal_train) * epoches
        N = len(self.datal_train.dataset)
        for epoch in range(current_epoch, epoches):
            self.model_enc.to(self.device)
            self.model_enc.train()
            self.model_dec.to(self.device)
            self.model_dec.train()
            self.channel.train()
            self.criterion.train()
            
            epoch_start = time.time()
            # Training
            re = self.train_one_epoch(n_iter, writer, snrs, max_iter, N)
            acc = self.validate(snrs)
            
            self.scheduler.step()
            epoch_end = time.time()
            time_for_one_epoch = epoch_end - epoch_start
            
            if save_model and ((epoch == 0) or (acc > best_acc)):
                best_acc = acc
                self.__save(epoch, path)
                
            if is_display:
                print('\n \n Time for one epoch',':', time_for_one_epoch)
                batches_start = time.time()
                print('[%d:%d]\t Losses:%.4f\t'% (epoch, len(self.datal_train), re.item()))
                print('ACC>>>>>:', acc)
   
        
    def train_one_epoch(self, n_iter, writer, snrs:list, max_iter, N):
        for i_batch, (x, labs) in enumerate(self.datal_train):
            x = x.to(self.device)
            labs = labs.to(self.device)
            num_imgs = x.shape[0]
            snrs = sample(snrs, num_imgs)
                
            z = self.model_enc(x, self.anchors)
            z = self.channel(z, snrs)
            out = self.model_dec(z)
            loss = self.__compute_loss(out, x,labs)
                
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.maxnorm > 0:
                torch.nn.utils.clip_grad_norm_(self.model_enc.parameters(), self.maxnorm)
                torch.nn.utils.clip_grad_norm_(self.model_dec.parameters(), self.maxnorm)
            self.optimizer.step()
            
            writer.add_scalar('train/losses', loss.item(), n_iter)
            n_iter += 1
            
        return loss
            
    
    def validate(self, snrs):
        acc = 0
        with torch.no_grad():
            self.model_enc.eval()
            self.model_dec.eval()

            for x, labs in self.datal_vali:
                x = x.to(self.device) 
                labs = labs.to(self.device)
                
                num_imgs = x.shape[0]
                snrs = sample(snrs, num_imgs)
                
                z_mean = self.model_enc(x, self.anchors)
                z = self.channel(z_mean, snrs)
                out = self.model_dec(z)
                acc += accuracy(out, labs)[0].item()
                
        return acc / len(self.datal_vali)
                
                
    def __compute_loss(self, out, x, labs):
        return self.criterion(out, labs)
                
    def __load(self, path=None):
        return load_model(self.model_enc, model_dec, self.optimizer, path)
       
    def __save(self, epoch, path=None):
        save_model(epoch, self.model_enc, self.model_dec, self.optimizer, path=path)  
        
    

def fit_alignment(z_arrived, target, aligner, 
                      criterion, optimizer, epoches, is_display=False):
    
    for epoch in range(epoches):
        out = aligner(z_arrived)
        loss = criterion(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if is_display and epoch % 100 == 0:
            print(loss.item())
            
def get_repr(enc_1, enc_2,
            anchors_x, anchors_y,
            channel_m, snrs):
    num_imgs = anchors_x.shape[0]
    snrs = sample(snrs, num_imgs)
    
    z_1 = enc_1(anchors_x)
    tilde_z_1 = channel_m(z_1, snrs)
    z_2 = enc_2(anchors_x)
    tilde_z_2 = channel_m(z_2, snrs)
    return z_1.detach(), z_2.detach(), tilde_z_1.detach(), tilde_z_2.detach()


class Aligner:
    
    def __init__(self, method: str, sc, dec=None):
        self.method = method
        if self.method == 'mlp':
            self.model = AlignerLinear(sc, sc)
        elif self.method == 'finetuning':
            assert dec is not None, "No decoder"
            self.model = AlignerNonLinear(dec, sc, sc)
            self.model.freeze_tail()
            
            
    def __snr2d2(self, snr):
        return 10**(-snr/10)
    
    def fit(self, z_arrived, z_target, y, config):
    
        if self.method == 'mlp':
            self.model.to(config.device)
            self.model.train()
            
            z_arrived = z_arrived.to(config.device)
            z_target  = z_target.to(config.device)
            criterion_base = nn.MSELoss()
            optimizer_base = optim.AdamW(params=self.model.parameters(), weight_decay=0.01, lr=config.lr)
            fit_alignment(z_arrived, z_target, self.model, criterion_base,
                              optimizer_base, config.epoches)
            self.model.eval()
            
        elif self.method == 'ls':
            self.model = ls(z_arrived, z_target).transpose(0, 1)
            self.model = self.model.to(config.device)

        elif self.method == 'mmse':
            self.model = mmse(z_arrived, z_target, self.__snr2d2(config.snr)).transpose(0,1)
            self.model = self.model.to(config.device)
            
        elif self.method == 'finetuning':
            self.model.to(config.device)
            
            z_arrived = z_arrived.to(config.device)
            y = y.to(config.device)
            criterion_base = nn.CrossEntropyLoss()
            optimizer_base = optim.AdamW(params=self.model.parameters(), weight_decay=0.01, lr=config.lr)
            fit_alignment(z_arrived, y, self.model, criterion_base,
                              optimizer_base, config.epoches)
            self.model.eval()
        else:
            assert 1 == 0, 'no such method'
            
    def transfrom(self, x):
        if self.method == 'mlp':
            return self.model(x)
        elif self.method == 'ls':
            return torch.matmul(
                x, self.model
            )
        elif self.method == 'mmse':
            return torch.matmul(
                x, self.model
            ) 
        elif self.method == 'finetuning':
            return self.model.aligner(x)
        else:
            assert 1 == 0, 'no such method'
        


def ls(z_arrived, z_target):
    """ z_target is the server side and noiseless """
    z_arrived, z_target = z_arrived.transpose(0,1), z_target.transpose(0,1)
    ZZT = torch.matmul(z_arrived, z_arrived.transpose(0, 1))
    ZZTinverse = torch.linalg.inv(ZZT)
    Z2ZT = torch.matmul(z_target, z_arrived.transpose(0,1))
    re_M =  torch.matmul(Z2ZT, ZZTinverse)
    return re_M

def mmse(z_arrived, z_target, delata2):
    """ z_target is the server side and noiseless, the delata2 is the variance of z_arrived """
    z_arrived, z_target = z_arrived.transpose(0,1), z_target.transpose(0,1)
    r = z_arrived.shape[1]
    n_t = z_arrived.shape[0]
    eyes = 2 * delata2 * n_t * torch.eye(r)
    eyes = eyes.to(z_arrived.device)
    ZTZI = torch.matmul(z_target.transpose(0,1), z_target) + eyes
    ZTZIinverse = torch.linalg.inv(ZTZI)
    re_M = torch.matmul(
        z_arrived, torch.matmul(ZTZIinverse, z_target.transpose(0, 1))
    )
    return re_M
    
        
    