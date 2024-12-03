import torch
import torch.nn as nn

from .modules import Resblock2d

  
    
    
class ENC_CIFAR10_MIXED(nn.Module):
    
    def __init__(self, symbol_channels):
        super().__init__()
        self.prep = nn.Sequential(
                    nn.Conv2d(3,64,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                    )
        self.layer1 = nn.Sequential(
                    nn.Conv2d(64,128,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.layer1_res = Resblock2d(128)
        
        self.layer2 = nn.Sequential(
                    nn.Conv2d(128,256,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2)
                    )
        self.layer3 = nn.Sequential(
                    nn.Conv2d(256,512,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False)
                    )
        self.encoder1 = nn.Sequential(
                        nn.Conv2d(512,4,kernel_size = 3,stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(4),
                        nn.ReLU()
                        )
        self.encoder2 = nn.Sequential(
                        nn.Linear(64,symbol_channels),
                        )
        
        self.model = nn.Sequential(
            self.prep,         # 64x32x32
            self.layer1,       # 128x16x16
            self.layer1_res,   # 256x16x16
            self.layer2,       # 256x8x8
            self.layer3,       # 512x4x4
            self.encoder1,     # 4x4x4
            nn.Flatten(),      # 64
            self.encoder2,     # sc
        )
        
        
    def forward(self, x):
        return self.model(x)
    
class DEC_CIFAR10_MIXED(nn.Module):
    
    def __init__(self, symbol_channels):
        super().__init__() 
        
        self.pre = nn.Sequential(
            nn.Linear(symbol_channels, 64), # 64
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.model = nn.Sequential(
            nn.Conv2d(4,512,kernel_size = 3,stride = 1, padding = 1, bias = False), # 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Resblock2d(512), # 512x4x4
            nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0, dilation = 1, ceil_mode = False), #512x1x1
            nn.Flatten(),
            nn.Linear(512,10,bias = False), #10
        )
        
    def forward(self, x):
        x = self.pre(x).reshape(-1, 4, 4, 4)
        return self.model(x) 
    
if __name__ == "__main__":
    # x = torch.randn(128, 1, 28, 28)
    # device = torch.device('cpu')
    # # enc = ENC_CIFAR10(8)
    # # dec = DEC_CIFAR10(8)
    
    # enc = ENC_MNIST(8)
    # dec = DEC_MNIST(8)
    
    # z = enc(x)
    # x_hat = dec(z)
    # print(z.shape)
    # print(x_hat.shape) 
    
    m_enc = ENC_CIFAR10_MIXED(16)
    m_dec = DEC_CIFAR10_MIXED(16)
    
    def summ(m):
        p_size = 0
        p_nele = 0
        for p in m.parameters():
            p_size += p.nelement() * p.element_size()
            p_nele += p.nelement()
        
        mb = p_size/ 1024 /1024
        print(p_nele, mb)
        
    summ(m_dec)