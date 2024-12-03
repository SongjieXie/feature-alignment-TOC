from torch import nn
import torch
from numbers import Number
from torch.nn import functional as F
import numpy as np
from torch import distributions as dist

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        dims = tuple((i for i in range(1, len(lpdf.shape), 1)))
        if reduce:
            return lpdf.sum(dim=dims)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Laplace(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Normal(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                            device=device)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1).to(device)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                               device=device)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1).to(device)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)


    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()

class Identity(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      
    def forward(self, x):
        return x
    
class ActivationLayer(nn.Module):
  def __init__(self, features: int) -> None:
    super().__init__()
    self.features = features
    self.hyper_block_scale = nn.Linear(1, self.features, bias=True)
    self.activation_fnc = Identity()
    
  def forward(self, inputs: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    scale = self.hyper_block_scale(betas)
    scale = self.activation_fnc(scale)
    if len(inputs.shape) == 4:
     # Unsqueeze for convolutional layers.
     scale = scale.unsqueeze(-1).unsqueeze(-1)
    return scale * inputs

class ActivatedLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True) -> None:
       super().__init__()
       self.pre = nn.Linear(in_channels, out_channels, bias=bias)
       self.act = ActivationLayer(1)
    
    def forward(self, x, betas):
        return self.act(self.pre(x), betas)
    
class ActivatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True) -> None:
       super().__init__()
       self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding, bias=bias)
       self.act = ActivationLayer(out_channels)
       
    def forward(self, x, betas):
        return self.act(self.pre(x), betas)
    
class ActivatedDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True) -> None:
       super().__init__()
       self.pre = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, output_padding=output_padding, bias=bias)
       self.act = ActivationLayer(out_channels)
       
    def forward(self, x, betas):
        return self.act(self.pre(x), betas)
    
class ActivatedConv1d(nn.Module):
    pass

class PreActActivatedResblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d_1(in_channel)
       self.norm_2 = nn.BatchNorm2d_2(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.activatedConv2d_1 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       self.activatedConv2d_2 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       
    def forward(self, x, betas):
        y = self.act_1(self.norm_1(x))
        y = self.activatedConv2d_1(y, betas)
        y = self.act_2(self.norm_2(y))
        y = self.activatedConv2d_2(y, betas)
        return x + y 
    
class ActivatedResblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d(in_channel)
       self.norm_2 = nn.BatchNorm2d(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.activatedConv2d_1 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       self.activatedConv2d_2 = ActivatedConv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       
    def forward(self, x, betas):
        y = self.activatedConv2d_1(x, betas)
        y = self.act_1(self.norm_1(y))
        y = self.activatedConv2d_2(y, betas)
        y = self.act_2(self.norm_2(y))
        return x + y 
    
class PreActResblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d(in_channel)
       self.norm_2 = nn.BatchNorm2d(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.Conv2d_1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       self.Conv2d_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
       
    def forward(self, x):
        y = self.act_1(self.norm_1(x))
        y = self.Conv2d_1(y)
        y = self.act_2(self.norm_2(y))
        y = self.Conv2d_2(y)
        return x + y 
    
class Resblock2d(nn.Module):
    def __init__(self, in_channel) -> None:
       super().__init__()
       self.norm_1 = nn.BatchNorm2d(in_channel)
       self.norm_2 = nn.BatchNorm2d(in_channel)
       self.act_1 = nn.ReLU()
       self.act_2 = nn.ReLU()
       self.Conv2d_1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       self.Conv2d_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False)
       
    def forward(self, x):
        y = self.Conv2d_1(x)
        y = self.act_1(self.norm_1(y))
        y = self.Conv2d_2(y)
        y = self.act_2(self.norm_2(y))
        return x + y 
    
class AFLayer(nn.Module):
    def __init__(self, channels) -> None:
       super().__init__()
       self.pool = nn.AdaptiveAvgPool2d((1,1))
       self.dense = nn.Sequential(
           nn.Linear(channels+1, channels//16),
           nn.ReLU(True),
           nn.Linear(channels//16, channels),
           nn.Sigmoid()
       )
    
    def forward(self, x, snr):
        y = self.pool(x)
        y = torch.squeeze(torch.squeeze(y, -1), -1)
        y = torch.cat((y, snr), -1)
        y = self.dense(y).unsqueeze(-1).unsqueeze(-1)
        return x*y

if __name__ == "__main__":
    # SNRs = torch.randn(128, 1)
    # x = torch.randn(128, 256, 4, 4)
    # m =AFLayer(256)
    # y = m(x, SNRs)
    # print(y.shape)
    
    inp = torch.randn(111, 128, 16, 16)
    model = Resblock2d(128)
    # model = PreActResblock2d(128)
    out = model(inp)
    print(out.shape)