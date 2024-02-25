import torch
import torch.nn as nn

def polynomial(Pdot, c=0, d=1):
    return (Pdot + c) ** d

def cosine(Pdot):
    return Pdot / torch.outer(torch.diag(Pdot), torch.diag(Pdot)) ** 0.5

def tanh(Pdot):
    return torch.tanh(Pdot)

def gaussian(Pdot, sigma=1):
    return torch.exp(-Pdot / (2 * sigma ** 2))

def laplacian(Pdot, sigma=1):
    return torch.exp(-Pdot / sigma)

def multi_kernel(Pdot, kernel_list=['cosine', 'gaussian', 'laplacian', 'polynomial', 'tanh']):
    Pdot_list = []
    for kernel in kernel_list:
        if kernel == 'cosine':
            Pdot_list.append(cosine(Pdot))
        elif kernel == 'gaussian':
            Pdot_list.append(gaussian(Pdot))
        elif kernel == 'laplacian':
            Pdot_list.append(laplacian(Pdot))
        elif kernel == 'polynomial':
            Pdot_list.append(polynomial(Pdot))
        elif kernel == 'tanh':
            Pdot_list.append(tanh(Pdot))
    
    Pdot = torch.stack(Pdot_list, dim=0).mean(dim=0)

    return Pdot

class KernelLayer(nn.Module):
    def __init__(self, n, kernel='laplacian'):
        super(KernelLayer, self).__init__()

        self.kernel = kernel
        self.weight = nn.Parameter(torch.ones(n, n), requires_grad=True)

    def forward(self, x):
        assert x.shape[0] == self.weight.shape[0], 'the size of x must be equal to the number of nodes'
        
        Pdot = torch.mm(x, x.t())

        if self.kernel == 'cosine':
            Pdot = cosine(Pdot)
        elif self.kernel == 'gaussian':
            Pdot = gaussian(Pdot)
        elif self.kernel == 'laplacian':
            Pdot = laplacian(Pdot)
        elif self.kernel == 'polynomial':
            Pdot = polynomial(Pdot)
        elif self.kernel == 'tanh':
            Pdot = tanh(Pdot)
        
        return Pdot * self.weight, Pdot
        
class MultiKernelLayer(nn.Module):
    def __init__(self, n, kernel_list=['cosine', 'gaussian', 'laplacian', 'polynomial', 'tanh']):
        super(MultiKernelLayer, self).__init__()

        self.kernel_list = kernel_list
        self.weight_list = nn.ParameterList([nn.Parameter(torch.ones(n, n), requires_grad=True) for _ in range(len(kernel_list))])

    def forward(self, x):
        
        Pdot = torch.mm(x, x.t())

        Pdot_list = []
        for weight, kernel in zip(self.weight_list, self.kernel_list):
            assert x.shape[0] == weight.shape[0], 'the size of x must be equal to the number of nodes'

            if kernel == 'cosine':
                Pdot_list.append(cosine(Pdot))
            elif kernel == 'gaussian':
                Pdot_list.append(gaussian(Pdot))
            elif kernel == 'laplacian':
                Pdot_list.append(laplacian(Pdot))
            elif kernel == 'polynomial':
                Pdot_list.append(polynomial(Pdot))
            elif kernel == 'tanh':
                Pdot_list.append(tanh(Pdot))
        
        Pdot = torch.stack(Pdot_list, dim=0).mean(dim=0)


        return Pdot, Pdot_list