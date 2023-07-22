import torch
import torch.nn as nn

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.xavier_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return nn.utils.spectral_norm(conv)

class SimpleSelfAttention(nn.Module):
    
    def __init__(self, n_in:int, ks=1, sym=False, tanh=False):#, n_out:int):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)      
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.sym = sym
        self.n_in = n_in
        self.tanh = tanh

    def forward(self, x, y = None):
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)

        if y is None:
            y = x

        size = x.size()  
        x = x.view(*size[:2],-1)   # (C,N)
        y = y.view(*size[:2],-1)
        
        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, y.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        o = self.gamma * o + x
        if self.tanh:
            o = torch.tanh(o)
        
        return o.view(*size).contiguous()       
