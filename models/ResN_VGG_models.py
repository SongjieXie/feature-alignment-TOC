import torch
import torch.nn as nn

class Bottleneck_enc(nn.Module):
    def __init__(self, input_dim, channel_sc):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, channel_sc)
        )
        
    def forward(self, X):
        return self.model(X)
 
 
class Bottleneck_dec(nn.Module):
    def __init__(self, symbol_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(symbol_channels, 64), # 64
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
    def forward(self, X):
        return self.model(X)  
    
    
class BasicBlock(nn.Module):
    """Basic Block for resnet

    """


    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
    
class ENC_CIFAR10(nn.Module):
    
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
        # self.layer1_res = Resblock2d(128)
        
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
        self.layer4 = nn.Sequential(
                        nn.Conv2d(512,4,kernel_size = 3,stride = 1, padding = 1, bias = False),
                        nn.BatchNorm2d(4),
                        nn.ReLU()
                        )
        self.enc = Bottleneck_enc(64, symbol_channels)
        
        self.model = nn.Sequential(
            self.prep,         # 64x32x32
            self.layer1,       # 128x16x16
            # self.layer1_res,   # 256x16x16
            self.layer2,       # 256x8x8
            self.layer3,       # 512x4x4
            self.layer4,     # 4x4x4
            self.enc     # sc
        )
        
        
    def forward(self, x):
        return self.model(x)
    
class DEC_CIFAR10(nn.Module):
    
    def __init__(self, symbol_channels):
        super().__init__() 
        
        self.pre = Bottleneck_dec(symbol_channels)
        self.model = nn.Sequential(
            nn.Conv2d(4,128,kernel_size = 3,stride = 1, padding = 1, bias = False), # 512x4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,512,kernel_size = 3,stride = 1, padding = 1, bias = False), # 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),# 512x4x4
            nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0, dilation = 1, ceil_mode = False), #512x1x1
            nn.Flatten(),
            nn.Linear(512,10,bias = False), #10
        )
        
    def forward(self, x):
        x = self.pre(x).reshape(-1, 4, 4, 4)
        return self.model(x) 
    
class ENC_ResNet_CIFAR10(nn.Module):
    
    def __init__(self, symbol_channels):
        super().__init__()
        self.prep = nn.Sequential(
                    nn.Conv2d(3,64,kernel_size = 3,stride = 1, padding = 1, bias = False),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
                    )
        self.layer1 = BasicBlock(64, 128, 2)
        # self.layer1_res = Resblock2d(128)
        
        self.layer2 = BasicBlock(128, 256, 2)
        self.layer3 = BasicBlock(256, 512, 2)
        self.layer4 = BasicBlock(512, 4, 1)
        self.enc = Bottleneck_enc(64, symbol_channels)
        
        self.model = nn.Sequential(
            self.prep,         # 64x32x32
            self.layer1,       # 128x16x16
            # self.layer1_res,   # 256x16x16
            self.layer2,       # 256x8x8
            self.layer3,       # 512x4x4
            self.layer4,     # 4x4x4
            self.enc     # sc
        )
        
        
    def forward(self, x):
        return self.model(x)
    
class DEC_ResNet_CIFAR10(nn.Module):
    
    def __init__(self, symbol_channels):
        super().__init__() 
        
        self.pre = Bottleneck_dec(symbol_channels)
        self.model = nn.Sequential(
            BasicBlock(4, 64, 1),
            BasicBlock(64, 128, 1), # 512x4x4
            BasicBlock(128, 512, 1), # 512x4x4
            nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0, dilation = 1, ceil_mode = False), #512x1x1
            nn.Flatten(),
            nn.Linear(512,10,bias = False), #10
        )
        
    def forward(self, x):
        x = self.pre(x).reshape(-1, 4, 4, 4)
        return self.model(x)     

    
    
if __name__ == "__main__":
    x = torch.randn(128, 3, 32, 32)
    # m_enc = ENC_ResNet_CIFAR10(16)
    # m_dec = DEC_ResNet_CIFAR10(16)
    
    m_enc = ENC_CIFAR10(16)
    m_dec = DEC_CIFAR10(16)
    
    def summ(m):
        p_size = 0
        p_nele = 0
        for p in m.parameters():
            p_size += p.nelement() * p.element_size()
            p_nele += p.nelement()
        
        mb = p_size/ 1024 /1024
        print(p_nele, mb)
        
    summ(m_dec)
    # print(sum([p.numel() for p in m_enc.parameters()]), "elements")
    # sc = m_enc(x)
    # print(sc.shape)
    # y = m_dec(sc)
    # print(y.shape)
    
    # mm = BasicBlock(3, 256, 2)
    # y = mm(x)
    # print(y.shape)
    
    
 
