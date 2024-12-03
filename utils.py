import torch
import numpy as np 
import os


from models.models import ENC_CIFAR10_MIXED, DEC_CIFAR10_MIXED
from models.ResN_VGG_models import ENC_CIFAR10, ENC_ResNet_CIFAR10, DEC_CIFAR10, DEC_ResNet_CIFAR10
from models.align_models import ZeroShotEncoder

from sklearn import datasets, decomposition
from dataloader import get_data


def get_models(model_name, SC, weight_path=None):
    if model_name == 'vgg':
        model_enc = ENC_CIFAR10(SC)
        model_dec = DEC_CIFAR10(SC)
    elif model_name == 'resnet':
        model_enc = ENC_ResNet_CIFAR10(SC)
        model_dec = DEC_ResNet_CIFAR10(SC)
    elif model_name == 'mixed':
        model_enc = ENC_CIFAR10_MIXED(SC)
        model_dec = DEC_CIFAR10_MIXED(SC)
    else:
        assert 1 == 0, "Unsupport !"
    if weight_path is not None:
        load_model(model_enc, model_dec, None, weight_path)
    return model_enc, model_dec

def get_models_with_on_device_alignment(model_name, SC, ratio, weight_path=None):
    model_enc = ZeroShotEncoder(model_name, SC)
    if model_name == 'vgg':
        model_dec = DEC_CIFAR10(SC*ratio)
    elif model_name == 'resnet':
        model_dec = DEC_ResNet_CIFAR10(SC*ratio)
    elif model_name == 'mixed':
        model_dec = DEC_CIFAR10_MIXED(SC*ratio)
    else:
        assert 1 == 0, "Unsupport !"
    if weight_path is not None:
        load_model(model_enc, model_dec, None, weight_path)
    return model_enc, model_dec


def get_path_2(config):
    name_1 = config.model + '-' +str(0) + '-' + config.channel\
        + '-' + str(config.snr) +'-'+ str(config.sc)
        
    name_2 = config.model + '-' +str(1) + '-' + config.channel\
        + '-' + str(config.snr) +'-'+ str(config.sc)
    
    if config.model == 'mixed':
        name_1 = 'CIFAR10' + '-' +str(0) + '-' + config.channel + '-' + str(config.snr) +'-'+ str(config.sc)
        name_2 = 'CIFAR10' + '-' +str(1) + '-' + config.channel + '-' + str(config.snr) +'-'+ str(config.sc)
        root = './local_pre'
        return os.path.join(root, name_1), os.path.join(root, name_2)
 
    return os.path.join(config.root, name_1), os.path.join(config.root, name_2)

def get_path(ID, DATA_SET, SNR, SC, ROOT):
    name = DATA_SET + '-' + str(ID) + '-' + 'awgn' + '-' + str(SNR) +'-'+ str(SC)
    path_to_backup = os.path.join(ROOT, name)
    return path_to_backup

def load_models(model_enc, model_dec, m_path):
    load_enc(model_enc, m_path)
    load_dec(model_dec, m_path)
    return model_enc, model_dec

""" Functions to use """

def obtain_models(id, data_set, SNR, SC, ROOT):
    enc, dec = get_models(data_set, SC)
    m_path = get_path(id, data_set, SNR, SC, ROOT)
    enc, dec = load_models(enc, dec, m_path)
    return enc, dec


""" previous """
def load_model(model_enc, model_dec, optimizer=None, path=None):
    path = '{0}/best.pt'.format(path)
    assert path is not None, "PATH !!!"
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        model_enc.load_state_dict(checkpoint['model_enc_states'])
        model_dec.load_state_dict(checkpoint['model_dec_states'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_states'])
        current_epoch = checkpoint['epoch']  
    else:
        assert 1 == 0, "No such ckp" + path
    return current_epoch

# def load_model(model_enc, model_dec, optimizer, path=None):
#     assert path is not None, "PATH !!!"
#     if os.path.isfile(path):
#         checkpoint = torch.load(path, map_location='cpu')
#         model_enc.load_state_dict(checkpoint['model_enc_states'])
#         model_dec.load_state_dict(checkpoint['model_dec_states'])
#         optimizer.load_state_dict(checkpoint['optimizer_states'])
#         current_epoch = checkpoint['epoch']  
#     return current_epoch

def save_model(epoch, model_enc, model_dec, optimizer, path=None):
    assert path is not None, "PATH !!!"
    with open('{0}/best.pt'.format(path), 'wb') as f:
        torch.save(
                    {
                    'epoch': epoch, 
                    'model_enc_states': model_enc.state_dict(), 
                    'model_dec_states': model_dec.state_dict(),
                    'optimizer_states': optimizer.state_dict(),
                    }, f
            ) 
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def psnr(img_true, img_fake):
    img_gen_numpy = img_fake.detach().cpu().float().numpy()
    # img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_numpy = (img_gen_numpy+1)/2.0 * 255.0
    # img_gen_numpy = img_gen_numpy * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8)

    origin_numpy = img_true.detach().cpu().float().numpy()
    # origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_numpy = (origin_numpy+1)/2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)

    diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2)

    PSNR = 10 * np.log10((255**2) / diff)
    return PSNR


def load_enc(model_enc, path=None):
    path = '{0}/best.pt'.format(path)
    assert path is not None, "PATH !!!"
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        model_enc.load_state_dict(checkpoint['model_enc_states'])
    else:
        assert 0 == 1, "NO PATH !!!"
    return model_enc

def load_dec(model_dec, path=None):
    path = '{0}/best.pt'.format(path)
    assert path is not None, "PATH !!!"
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        model_dec.load_state_dict(checkpoint['model_dec_states'])
    else:
        assert 0 == 1, "NO PATH !!!"
    return model_dec
        
def evaluate(model_enc, model_dec, datal_vali,
             channel_m, snrs, device):
    acc = 0
    with torch.no_grad():
        model_enc.to(device)
        model_dec.to(device)
        model_enc.eval()
        model_dec.eval()
        acc = 0
        for x, labs in datal_vali:
            x = x.to(device)
            labs = labs.to(device)
            
            num_imgs = x.shape[0]
            snrs = sample(snrs, num_imgs)
            
            z_mean = model_enc(x)
            z = channel_m(z_mean, snrs)
            out = model_dec(z)
            acc += accuracy(out, labs)[0].item()
                
        return acc / len(datal_vali)
    
def evaluate_relative(model_enc, model_dec, datal_vali,
             channel_m, snrs, device, anchors):
    acc = 0
    with torch.no_grad():
        model_enc.to(device)
        model_dec.to(device)
        model_enc.eval()
        model_dec.eval()
        acc = 0
        for x, labs in datal_vali:
            x = x.to(device)
            labs = labs.to(device)
            
            num_imgs = x.shape[0]
            snrs = sample(snrs, num_imgs)
            
            z_mean = model_enc(x, anchors)
            z = channel_m(z_mean, snrs)
            out = model_dec(z)
            acc += accuracy(out, labs)[0].item()
                
        return acc / len(datal_vali)
    
    
def evaluate_with_alignment(model_enc, model_dec, aligner, 
                            datal_vali, channel_m, snrs, device):
    acc = 0
    with torch.no_grad():
        model_enc.to(device)
        model_dec.to(device)
        model_enc.eval()
        model_dec.eval()
        acc = 0
        for x, labs in datal_vali:
            x = x.to(device)
            labs  = labs.float() 
            labs = labs.to(device)
            
            num_imgs = x.shape[0]
            snrs = sample(snrs, num_imgs)
    
            
            z_mean = model_enc(x)
            z = channel_m(z_mean, snrs)
            z = aligner.transfrom(z) # Semantic alignment
            out = model_dec(z)
            acc += accuracy(out, labs)[0].item()
                
        return acc / len(datal_vali)
    
    
    
    
    
def sample(d_range: list, num_pts: int, sampling='uniform'):
    if sampling == 'uniform':
        return (d_range[1]-d_range[0])*torch.rand(num_pts, 1) + d_range[0]
    elif sampling == 'discrete':
        return torch.randint(1, 21, size=(num_pts, 1)).float()
    else:
        pl = torch.empty(num_pts, 1)
        return nn.init.trunc_normal_(pl, 10, float(sampling), 0, 20)
    
    
    
""" Anchors """
def creat_anchors(args):
    dataloader_tmp = get_data(args.dataset, args.sc*args.ratio, 
                               n_worker= args.num_workers, 
                               train=True)
    
    anchors, _ = next(iter(dataloader_tmp))
    
    while len(set(_.tolist())) < 10:
        anchors, _ = next(iter(dataloader_tmp))
        
    
    # print(set(_.tolist()))
    # print(anchors.shape)
    return anchors

def save_anchors(anchors, args):
    with open('{0}_anchors.pt'.format(args.anchors_path), 'wb') as f:
            torch.save(
                    {
                    'anchors': anchors, 
                    }, f
            ) 

def load_anchors(args):
    anchors_path = '{0}_anchors.pt'.format(args.anchors_path)
    if os.path.isfile(anchors_path):
        print('Found saved anchor data, loading ...')
        checkpoint = torch.load(anchors_path, map_location='cpu')
        anchors = checkpoint['anchors']
    else:
        print('Creating Anchor data ...')
        anchors = creat_anchors(args)
        save_anchors(anchors, args)
    print('Done. The number of anchors: {0}'.format(anchors.shape[0]))
    return anchors
    
    
    
    
def pca_reduce(features):
    pca = decomposition.PCA(n_components=2)
    pca.fit(features)
    return pca.transform(features)

def draw_sactters(ax, emb, labs, method_title):
    # assert len(emb) == len(labs)
    list_labs = np.array(
        list(set(labs)))
    i = 0
    for lab in list_labs:
        i += 1
        if i == 6:
            idxes_where = np.where(labs==lab)
            ax.scatter(emb[idxes_where][:,0], emb[idxes_where][:,1],
                 s=8, label=lab, alpha=1, lw=2, c='r')
        elif i ==2:
            idxes_where = np.where(labs==lab)
            ax.scatter(emb[idxes_where][:,0], emb[idxes_where][:,1],
                 s=8, label=lab, alpha=1, lw=2, c='b')
            
        elif i ==1:
            idxes_where = np.where(labs==lab)
            ax.scatter(emb[idxes_where][:,0], emb[idxes_where][:,1],
                 s=8, label=lab, alpha=1, lw=2, c='g')
        else:
            idxes_where = np.where(labs==lab)
            ax.scatter(emb[idxes_where][:,0], emb[idxes_where][:,1],
                    s=8, label=lab, alpha=0.1, lw=2)
    # ax.set_title(method_title, fontsize=15)
    # ax.grid(linestyle='-', axis='both')
    # ax.legend()
    
    