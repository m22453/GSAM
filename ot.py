import torch
from tqdm import tqdm
from torch.nn import CrossEntropyLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cost_matrix(x, y):
    """
    return cost matrix (cosine distance) C (n, k)
    """
    D = x.size(0)
    x = x.view(D, -1)
    
    x = x.div(x.norm(p=2, dim=1, keepdim=True) + 1e-12)
    y = y.div(y.norm(p=2, dim=1, keepdim=True) + 1e-12)
    cos_dis = torch.mm(x, y.t())
    return 1 - cos_dis, cos_dis


def IPOT(C, miu, nu, beta=0.5):
    """
    C: (n, k) cost matrix
    n: int, number of samples in X
    k: int, number of samples in Y
    miu: (n, 1)
    nu: (k, 1)
    """
    n, k = C.shape
    # sigma = torch.ones((n, 1)).float().to(device) / n
    sigma = torch.ones((k, 1)).float().to(device) / k
    miu = miu.to(device)
    nu = nu.to(device)

    T = torch.ones((n, k)).float().to(device)
    C = torch.exp(-C / beta).float().to(device)
    for _ in tqdm(range(20)):
        T = C * T  # element-wise product (n, k)
        for _ in range(1):
            delta = miu / torch.squeeze(torch.matmul(T, sigma))
            # sigma = torch.unsqueeze(nu / torch.squeeze(torch.matmul(torch.transpose(T, 0, 1), delta)), 1)
            sigma = torch.unsqueeze(nu, 1) / torch.matmul(torch.transpose(T, 0, 1), torch.unsqueeze(delta, 1))
        tmp = torch.unsqueeze(delta, 1) * T * sigma.transpose(1, 0)
        if torch.isnan(T).any():
            print('T is nan')
            break
        else:
            T = tmp
    # print(T.shape)
    return T.detach().softmax(dim=1)

def IPOT_distance(C, miu, nu):
    C = C.float().to(device)
    T = IPOT(C, miu, nu)
    # print(torch.min(T), torch.max(T))
    distance = torch.trace(torch.mm(torch.transpose(C, 0, 1), T))
    return -distance, T

def cross_att(X, Y):
    """
    X: (n, d)
    Y: (n, d)
    """
    _, S = cost_matrix(X, Y)
    Sii = torch.diag(S).contiguous().view(-1, 1)
    att_X = Sii * X
    att_Y = Sii * Y
    return att_X, att_Y


def get_ot_loss(X, Y, miu, nu):
    """
    X: (n, d)
    Y: (k, d)
    miu: (n, 1)
    nu: (k, 1)
    """
    C, _ = cost_matrix(X, Y)
    d, T = IPOT_distance(C, miu, nu)
    return d, T

def get_max_code(T, codebook):
    """
    T: (n, d)
    codebook: (d, k)
    """
    T_max = torch.argmax(T, dim=1, keepdim=True)
    assert T_max.shape == T.shape, 'the shape is changed'
    return torch.mm(T_max, codebook)

# n = 10
# k = 3
# d = 128

# X = torch.randn((n, d)).float().to(device)
# Y = torch.randn((k, d)).float().to(device)

# C, S = cost_matrix(X, Y)

# print(C.shape)

# u = torch.randn(n, requires_grad=True)
# v = torch.randn(k, requires_grad=True)

# optimizer = torch.optim.Adam([u, v], lr=0.01)

# # number of iteartions
# n_iter = 200

# for i in range(n_iter):

#     d, T = IPOT_distance(C, u, v)   

#     if i % 10 == 0:
#         print(f"i = {i}, d = {d}")
    
#     d.backward()
#     optimizer.step()
#     optimizer.zero_grad()


# x2y = torch.mm(X, Y.t()).softmax(dim=1) 
# print(f"x2y = {x2y}")
# assert torch.sum(x2y, dim=1).allclose(torch.ones(n).to(device)), "x2y is not a probability distribution"

# x2y = x2y / 0.05
# loss = CrossEntropyLoss()
# print(loss(x2y, T))



