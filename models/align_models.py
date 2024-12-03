import torch
import torch.nn as nn

from .models import ENC_CIFAR10_MIXED, DEC_CIFAR10_MIXED
from .ResN_VGG_models import ENC_CIFAR10, ENC_ResNet_CIFAR10, DEC_CIFAR10, DEC_ResNet_CIFAR10


class AlignerLinear(nn.Module):
    
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.model = nn.Linear(input_dim, out_dim, bias=False)
        
    def forward(self, x):
        return self.model(x)
    
    
class AlignerLinearConv(nn.Module):
    
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Linear(input_dim, out_dim)
        
    def forward(self, x):
        x_shape = x.shape
        y = self.model(self.flatten(x))
        return y.view(x_shape)
    
    
class AlignerNonLinear(nn.Module):
    
    def __init__(self, tail_model, input_dim, out_dim,
                 hidden_size=8):
        super().__init__()
        self.aligner = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim, bias=False)
        )
        self.tail_model = tail_model
        
    def freeze_tail(self):
        
        for n, p in self.named_parameters():
            if 'aligner' not in n:
                p.requires_grad = False
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            
    def forward(self, x):
        x = self.aligner(x)
        return self.tail_model(x)
    
    
class ZeroShotEncoder(nn.Module):
    
    def __init__(self, model_name, sc):
        super().__init__()
        self.model = self.__build_model(model_name, sc)
        
    def __build_model(self, model_name, SC):
        if model_name == 'vgg':
            model_enc = ENC_CIFAR10(SC)
        elif model_name == 'resnet':
            model_enc = ENC_ResNet_CIFAR10(SC)
        elif model_name == 'mixed':
            model_enc = ENC_CIFAR10_MIXED(SC)
        else:
            assert 1 == 0, "Unsupport !"
        return model_enc
    
    def forward(self, x, anchors):
        Z_x = self.model(x)
        Z_anchors = self.model(anchors)
        Z_rel = sim_matrix(Z_x, Z_anchors)
        return Z_rel
    
def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
    

