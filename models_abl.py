import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function
from typing import Any, Optional, Tuple

from kernel_opts import KernelLayer, MultiKernelLayer


from task_loss import CGDDCLoss
from prototype_loss import PtypeLoss
from contrastive_loss import ClusterLoss, Loss, InstanceLoss

class Encoder(nn.Module):
    def __init__(self, n_input, n_z, dropout=0.0):
        super(Encoder, self).__init__()

        n_enc_1, n_enc_2, n_enc_3 = 500, 500, 2000

        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_enc_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_enc_1, n_enc_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_enc_2, n_enc_3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_enc_3, n_z),
        )
     
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self,  n_input, n_z, dropout=0.0):
        super(Decoder, self).__init__()

        n_dec_1, n_dec_2, n_dec_3 = 2000, 500, 500

        self.decoder = nn.Sequential(
            nn.Linear(n_z, n_dec_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_dec_1, n_dec_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_dec_2, n_dec_3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_dec_3, n_input),
        )
    def forward(self, x):
        return self.decoder(x)


class AE(nn.Module):
    def __init__(self, n_input, n_z, dropout=0.0):
        super(AE, self).__init__()

        self.encoder = Encoder(n_input, n_z, dropout)
        self.decoder = Decoder(n_input, n_z, dropout)

    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return z, x_

class ConMultiAE(nn.Module): # for checking the performance
    def __init__(self, views:int, input_sizes:list, n_z:int, shared=False, dropout=0.0):
        super(ConMultiAE, self).__init__()

        self.encoders = []
        self.decoders = []

        if shared:
            assert sum(input_sizes) == input_sizes[0] * views, "input size is not the same"

            v = 0
            self.encoders.append(Encoder(input_sizes[v], n_z, dropout))
            self.decoders.append(Decoder(input_sizes[v], n_z, dropout))

        else:
            for v in views:
                self.encoders.append(Encoder(input_sizes[v], n_z, dropout))
                self.decoders.append(Decoder(input_sizes[v], n_z, dropout))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.views = views
        self.shared = shared

    def forward(self, xs):
        # encoding
        zs = []
        for v in range(self.views):
            if self.shared:
                v= 0 
            z = self.encoders[v](xs[v])
            zs.append(z)
        

        # decoding
        xs_ = []
        for v in range(self.views):
            if self.shared:
                v= 0
            x_ = self.decoders[v](zs[v])
            xs_.append(x_)
        
        return zs, xs_

        

class DualEncoder(nn.Module): # for checking the performance
    def __init__(self, n_input, n_z, dropout=0.0):
        super(DualEncoder, self).__init__()

        n_enc_1, n_enc_2, n_enc_3 = 500, 500, 2000

        self.uni_encoder = nn.Sequential(
            nn.Linear(n_input, n_enc_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_enc_1, n_enc_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_enc_2, n_enc_3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_enc_3, n_z),
        )
        self.con_encoder_1 = nn.Sequential(
            nn.Linear(n_input, n_enc_1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.con_encoder_2 = nn.Sequential(
            nn.Linear(n_enc_1, n_enc_2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.con_encoder_3 = nn.Sequential(
            nn.Linear(n_enc_2, n_enc_3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.con_nz = nn.Linear(n_enc_3, n_z)
    
    def forward(self, x):
        con_1_x = self.con_encoder_1(x)
        con_2_x = self.con_encoder_2(con_1_x)
        con_3_x = self.con_encoder_3(con_2_x)
        con_z = self.con_nz(con_3_x)

        return self.uni_encoder(x), (con_1_x, con_2_x, con_3_x, con_z)

class VectorQuantizer(nn.Module):
    """
    discrete latent variable

    """
    def __init__(self, code_dim, code_num, beta, alpha=1.0):
        super(VectorQuantizer, self).__init__()
        
        self.code_num = code_num
        self.code_dim = code_dim
        self.beta = beta
        self.alpha = alpha

        # self.codebook = nn.Parameter(torch.randn(code_num, code_dim))
        # torch.nn.init.xavier_normal_(self.codebook.data)

        self.codebook = nn.Embedding(code_num, code_dim)
        # self.codebook.weight.data.uniform_(-1.0 / self.code_num, 1.0 / self.code_num)
        torch.nn.init.xavier_normal_(self.codebook.weight.data)        

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, d_dim)

        quantization pipeline:

            get encoder input (B,D) and codebook (C,D)

        """

        # calculate distances between z and codebook
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.codebook.weight, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()  # Make sure each sample's n_values add up to 1.

        # distances from z to embeddings e_j 
        # (z - e)^2 = z^2 + e^2 - 2 e * z

        # z_rec = torch.matmul(q, self.codebook.weight)

        # return q, z_rec

        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - 2 * \
            torch.matmul(z, self.codebook.weight.t())
        
        # find closest code 
        max_indices = torch.argmax(d, dim=1).unsqueeze(1)
        max_probs = torch.zeros(max_indices.shape[0], self.code_num).to(z.device)
        max_probs.scatter_(1, max_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(max_probs, self.codebook.weight)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z)**2) +  torch.mean((z_q - z.detach())**2)

        # straight through estimator
        z_q = z + (z_q - z).detach()

        # get perplexity
        e_mean = torch.mean(max_probs, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match input shape
        z_q = z_q.view(z.shape)

        return q, z_q, loss #, perplexity, max_indices, max_probs
    

def central_moments(data, order):
    mean = torch.mean(data, dim=0)
    moments = []
    for i in range(1, order + 1):
        if i == 1:
            moments.append(torch.zeros_like(mean))  # 第一中心矩总是0
        else:
            moments.append(torch.mean((data - mean) ** i, dim=0))
    return moments

def central_moment_discrepancy(data1, data2, order):
    moments1 = central_moments(data1, order)
    moments2 = central_moments(data2, order)
    
    discrepancy = 0
    for m1, m2 in zip(moments1, moments2):
        discrepancy += torch.sum((m1 - m2) ** 2)
    return discrepancy


        
def target_distribution(q):
    p = q ** 2 / q.sum(0)
    p = (p.T / p.sum(1)).T
    return p

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BTNet(nn.Module):
    def __init__(self, sizes, lambd=0.0051) -> None:
        super(BTNet, self).__init__()

        self.lambd = lambd
        # layers = []
        # for i in range(len(sizes) - 2):
        #     layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
        #     layers.append(nn.BatchNorm1d(sizes[i + 1]))
        #     layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        # self.projector = nn.Sequential(*layers)
        in_dim, hidden_dim = sizes[0], sizes[1]
        self.projector =  nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def forward(self, x1, x2):
        
        batch_size = x1.shape[0]

        z1 = self.projector(x1)
        z2 = self.projector(x2)

        # z1 = x1
        # z2 = x2

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        # c = F.normalize(z1, dim=1, p=2).T @ F.normalize(z2, dim=1, p=2)

        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


class InsNet(nn.Module):
    def __init__(self, sizes) -> None:
        super(InsNet, self).__init__()

        in_dim, hidden_dim = sizes[0], sizes[1]
        self.projector =  nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # normalization layer for the representations z1 and z2
        self.ln = nn.LayerNorm(hidden_dim)
        self.ins = InstanceLoss()

    def forward(self, x1, x2):
        
        z1 = self.projector(x1)
        z2 = self.projector(x2)

        # z1 = self.ln(z1)
        # z2 = self.ln(z2)

        z1 = F.normalize(z1, dim=1, p=2)
        z2 = F.normalize(z2, dim=1, p=2)

        loss = self.ins(z1, z2)
        
        return loss


class MutualAlign(nn.Module):
    def __init__(self, n_z, hidden_dim=128, beta=0.00005, gamma=0.0001):
        super(MutualAlign, self).__init__()

        self.fea_sim_layer = BTNet([n_z, hidden_dim])
        self.ins_sim_layer = InsNet([n_z, hidden_dim])
        self.beta = beta  # for feature alignment
        self.gamma = gamma # for instance alignment
    
    def forward(self, z1, z2):
        loss = self.fea_sim_layer(z1, z2) * self.beta + self.ins_sim_layer(z1, z2) * self.gamma
        return loss


class PrototypeNet(nn.Module):
    def __init__(self, n_z, hidden_dim, n_prototype, epsilon=0.05, sinkhorn_iterations=3, temperature=0.2):
        super(PrototypeNet, self).__init__()

        # hyparameters
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.temperature = temperature
        self.n_prototype = n_prototype

        # get the mapping from z to prototype
        self.linear = nn.Sequential(
            nn.Linear(n_z, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_prototype),
        )

        self.prototype_layer = nn.Linear(hidden_dim, n_prototype, bias=False)
        

        self.get_assignments = self.sinkhorn

        self.ddc_loss = CGDDCLoss(num_cluster=n_prototype)

    def sinkhorn(self, Q, nmb_iters=3):
        ''' 
            :param Q: (num_prototypes, batch size)
        '''
        
        with torch.no_grad():
            # make the matrix sums to 1
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            u = torch.zeros(K).to(Q.device)
            r = torch.ones(K).to(Q.device) / K
            c = torch.ones(B).to(Q.device) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
        



    def vq_loss(self, proto_out, q, beta=0.25):
        # find closest code by assignment
        max_indices = torch.argmax(q, dim=1).unsqueeze(1)
        max_probs = torch.zeros(max_indices.shape[0], self.n_prototype).to(q.device)
        max_probs.scatter_(1, max_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(max_probs, self.prototype_layer.weight) # (batch, n_z)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - proto_out)**2) +  torch.mean((z_q - proto_out.detach())**2) * beta

        return loss

    def forward(self, z1, z2):
        
        z1 = self.linear(z1)
        z1 = F.normalize(z1, dim=1, p=2)
        
        z2 = self.linear(z2)
        z2 = F.normalize(z2, dim=1, p=2)

        z_fusion = z1 + z2

        # normalize prototype layer
        with torch.no_grad():
            w = self.prototype_layer.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototype_layer.weight.copy_(w)

        # compute assign code of z1 and z2
        proto_out_1 = self.prototype_layer(z1)
        proto_out_2 = self.prototype_layer(z2)


        # TODO: define this to hparams
        with torch.no_grad():
            q1 = torch.exp(
                proto_out_1 / self.epsilon).t()
            q1 = self.get_assignments(
                q1, self.sinkhorn_iterations)    
            q2 = torch.exp(
                proto_out_2 / self.epsilon).t()
            q2 = self.get_assignments(
                q2, self.sinkhorn_iterations)   
            
            
        p1 = F.softmax(q1 / self.temperature, dim=1)
        p2 = F.softmax(q2 / self.temperature, dim=1)

        loss_1t2_proto = torch.mean(torch.sum(-q1 * torch.log(p2 + 1e-8), dim=1))
        loss_2t1_proto = torch.mean(torch.sum(-q2 * torch.log(p1 + 1e-8), dim=1))

        loss_proto = (loss_1t2_proto + loss_2t1_proto) / 2.0

        loss_vq = self.vq_loss(z1, q1) + self.vq_loss(z2, q2)


        return (loss_proto, loss_vq)


class GradReverse(Function):        
    """
        重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


def grad_reverse(x, coeff=1):
    return GradReverse.apply(x, coeff)

class ViewClassifier(nn.Module):
    def __init__(self, n_z, hidden_dim=200):
        super(ViewClassifier, self).__init__()
        self.fc1 = nn.Linear(n_z, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = grad_reverse(x)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)


class CrossModal(nn.Module):
    def __init__(self,  n_input_m, n_input_a, n_z, code_num, n_view=2, normlize=True):
        super(CrossModal, self).__init__()

        self.alpha = 1.0

        n_enc_1, n_enc_2, n_enc_3 = 500, 500, 2000

        self.dual_encoder_main = DualEncoder(n_input_m, n_z)
        self.dual_encoder_aux = DualEncoder(n_input_a, n_z)

        # self.mutual_align = MutualAlign(n_z)
        # self.mutual_align = MutualAlign(n_enc_1) # for 1st layer
        # self.mutual_align = MutualAlign(n_enc_2) # for 2nd layer
        self.mutual_align = MutualAlign(n_enc_3) # for 3rd layer

        self.decoder_main = Decoder(n_input_m, n_z)
        self.decoder_aux = Decoder(n_input_a, n_z)

        # self.encoder_main = AE(n_input_m, n_z)

        # pass continuous latent vector through discretization bottleneck
        self.quantizer = VectorQuantizer(n_z, code_num, beta=0.25)

        feature_dim = 128
        # ensure the same dimension
        self.specific_channel = nn.Sequential(
            nn.Linear(n_z, feature_dim),
            # nn.ReLU(),
            # nn.Linear(feature_dim, feature_dim),
            
        )
        self.common_channel = nn.Sequential(
            nn.Linear(n_z * n_view, feature_dim),
            # nn.ReLU(),
            # nn.Linear(feature_dim, feature_dim),
        )
        # self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=n_z* n_view, nhead=1, dim_feedforward=256)

        self.TransformerEncoderLayer = TransformerEncoderLayerWithLayerScale(n_z* n_view, nhead=1, dim_feedforward=256)

        self.contrastive_loss = Loss()
        # self.ddc = CGDDCLoss(code_num)
        self.ddc = PtypeLoss(code_num)
        self.cel = nn.CrossEntropyLoss()

        self.codebook = nn.Linear(n_z, code_num, bias=False)


        self.normlize = normlize
        self.code_num = code_num


    def forward(self, x_main, x_aux, type='warm_up'):
        
        # z1_m, (z2_m_i, z2_m) = self.dual_encoder_main(x_main)
        # z1_a, (z2_a_i, z2_a) = self.dual_encoder_aux(x_aux)
        z1_m, (con_1_m, con_2_m, con_3_m, z2_m) = self.dual_encoder_main(x_main)
        z1_a, (con_1_a, con_2_a, con_3_a, z2_a) = self.dual_encoder_aux(x_aux)


        q_m, z2_m_rec, loss_m = self.quantizer(z2_m)
        q_a, z2_a_rec, loss_a = self.quantizer(z2_a)


        x_main_hat = self.decoder_main(z1_m) 
        x_aux_hat = self.decoder_aux(z1_a)

        x_main_hat_ = self.decoder_main(z2_m) 
        x_aux_hat_ = self.decoder_aux(z2_a)

        # fusion and contrastive 

        # # h_2_m_nor = F.normalize(self.specific_view(z2_m), dim=1)
        # # h_2_a_nor = F.normalize(self.specific_view(z2_a), dim=1)

        # h_2_m_nor = self.specific_view(z2_m)
        # h_2_a_nor = self.specific_view(z2_a)

        # h_2, S = self.view_fusion(z2_m, z2_a)

        # # h_2_nor = F.normalize(h_2, dim=1)
        # h_2_nor = h_2

        # fusion and contrastive 
        
        if self.normlize:

            h_2_m_nor = F.normalize(self.specific_channel(z2_m), dim=1)
            h_2_a_nor = F.normalize(self.specific_channel(z2_a), dim=1)


            h_2, S = self.view_fusion(z2_m, z2_a)

            h_2_nor = F.normalize(h_2, dim=1)

        else:
            h_2_m_nor = self.specific_channel(z2_m)
            h_2_a_nor = self.specific_channel(z2_a)


            h_2, S = self.view_fusion(z2_m, z2_a)

            h_2_nor = h_2
        

        if type == 'warm_up':
            loss_rec = F.mse_loss(x_main_hat, x_main) + F.mse_loss(x_aux_hat, x_aux) + F.mse_loss(x_main_hat_, x_main) + F.mse_loss(x_aux_hat_, x_aux)
        
        elif type == 'mutual_align':
            
            loss_rec =  F.mse_loss(x_main_hat_, x_main) + F.mse_loss(x_aux_hat_, x_aux)
            # loss_rec += self.mutual_align(z2_m, z2_a)  # hidden  layer
            # loss_rec += self.mutual_align(con_1_m, con_1_a) # 1st layer
            # loss_rec += self.mutual_align(con_2_m, con_2_a) # 2nd layer
            loss_rec += self.mutual_align(con_3_m, con_3_a) # 3rd layer


        elif type == 'fusion_align':
            
            loss_rec = F.mse_loss(x_main_hat_, x_main) + F.mse_loss(x_aux_hat_, x_aux) 

            # for assignments
            zs = torch.add(z2_a, z2_m)
            if z2_a.shape == h_2.shape:
                zs = zs + h_2 # skipping connection

            q_m, _ = self.get_qp(z2_m)
            q_a, _ = self.get_qp(z2_a)
            q, _ = self.get_qp(zs)

            # # cmd
            # order = 4
            # loss_cmd = central_moment_discrepancy(h_2_nor, h_2_m_nor, order) + central_moment_discrepancy(h_2_nor, h_2_a_nor, order)

            # print(f"loss_cmd: {loss_cmd}")

            # for feature
            # loss_cons = self.fusion_align_fea_loss(h_2_nor, h_2_m_nor, h_2_a_nor, S)
            loss_cons = self.fusion_align_fea_loss(q, q_m, q_a, S)

            # TODO: define this to hparams
            loss_rec += (loss_cons) * 0.0

            thr =  1.0 / self.code_num

            loss_ass = self.fusion_align_assign_loss(zs, z2_m, z2_a, h_2)

            # loss_stru = self.fusion_align_structure_loss(zs, z2_m, z2_a, thr)

            loss_rec += loss_ass 
            # loss_rec += loss_stru

            
            # print(f"loss_stru: {loss_stru}")
        
        return (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat, q_m, q_a, h_2_nor, loss_rec
    
    def view_fusion(self, z2_m, z2_a):

        z_s = torch.cat([z2_m, z2_a], dim=1)

        commonz, S = self.TransformerEncoderLayer(z_s)
        commonz = self.common_channel(commonz) # (b, n_z)

        return commonz, S
    
    def get_qp(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.codebook.weight, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()  # Make sure each sample's n_values add up to 1. 

        p = target_distribution(q)
        return (q, p)
    
    def fusion_align_fea_loss(self, h_2_nor, h_2_m_nor, h_2_a_nor, S, beta=0.0001):
        
        loss_cons = self.contrastive_loss(h_2_nor, h_2_m_nor, S) + self.contrastive_loss(h_2_nor, h_2_a_nor, S)  
        return loss_cons * beta
    
    def fusion_align_assign_loss(self, zs, z2_m, z2_a, fused_z=None, omega=0.01):
        q_m, _ = self.get_qp(z2_m)
        q_a, _ = self.get_qp(z2_a)
        q, _ = self.get_qp(zs)

        if fused_z is None:
            fused_z = zs
            
        loss_ass = 0.1 * (self.ddc(q_m, fused_z) + self.ddc(q_a, fused_z)) + self.ddc(q, zs)
        return loss_ass * omega
    
    def fusion_align_structure_loss(self, zs, z2_m, z2_a, thr, theta=0.0001):

        q, _ = self.get_qp(zs)

        # target logits
        p_sim= torch.matmul(q, q.t())
        p_sim[p_sim < thr] = 0
        p_sim = F.normalize(p_sim, p=2, dim=1)

        def mm_normalize(z):
            sim = torch.matmul(z, z.t())
            sim = F.normalize(sim, p=2, dim=1)
            return sim

        z2_m_sim, z2_a_sim = mm_normalize(z2_m), mm_normalize(z2_a)

        loss = self.cel(z2_m_sim, p_sim) + self.cel(z2_a_sim, p_sim)

        return loss * theta
        

class DecomposeNetwork(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input_m, n_input_a, n_z, code_num):
        super(DecomposeNetwork, self).__init__()

        self.input_m_layer = nn.Linear(n_input_m, n_enc_1)
        self.input_a_layer = nn.Linear(n_input_a, n_enc_1)

        self.encoder_main = nn.Sequential(
                                    nn.Linear(n_enc_1, n_enc_2),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_2, n_enc_3),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_3, n_z),
                                )
        self.encoder_aux = nn.Sequential(
                                    nn.Linear(n_enc_1, n_enc_2),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_2, n_enc_3),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_3, n_z),
                                )
        self.encoder_shared = nn.Sequential(
                                    nn.Linear(n_enc_1, n_enc_2),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_2, n_enc_3),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_3, n_z),
                                )
        
        # pass continuous latent vector through discretization bottleneck
        self.quantizer = VectorQuantizer(n_z, code_num, beta=0.25)
        
        self.decoder_shared = nn.Sequential(
                                    nn.Linear(n_z, n_dec_1),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_1, n_dec_2),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_2, n_dec_3),
                                    nn.ReLU(),
                                )
        self.decoder_m_layer = nn.Linear(n_dec_3, n_input_m)
        self.decoder_a_layer = nn.Linear(n_dec_3, n_input_a)


    def forward(self, x_m, x_a):

        x_m = F.relu(self.input_m_layer(x_m)) 
        x_a = F.relu(self.input_a_layer(x_a))

        z_m_p = self.encoder_main(x_m) # parity part
        z_a_p = self.encoder_aux(x_a)

        z_m_s = self.encoder_shared(x_m)
        z_a_s = self.encoder_shared(x_a)

        q_m, z_m_s_c, loss_m = self.quantizer(z_m_s) # the shared part
        q_a, z_a_s_c, loss_a = self.quantizer(z_a_s)
        
        # z_m = z_m_p + z_m_s_c
        # z_a = z_a_p + z_a_s_c

        z_m = z_m_p + z_m_s
        z_a = z_a_p + z_a_s

        loss_code = loss_m + loss_a

        x_m_hat = self.decoder_m_layer(self.decoder_shared(z_m))
        x_a_hat = self.decoder_a_layer(self.decoder_shared(z_a))

        return (z_m_p, z_m_s, z_a_p, z_a_s), (x_m_hat, x_a_hat), loss_code
    
class CrossViewModel(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input_m, n_input_a, n_z, code_num):
        super(CrossViewModel, self).__init__()

        self.d_net = DecomposeNetwork(n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input_m, n_input_a, n_z, code_num)

        self.view_cls = ViewClassifier(n_z)


    def forward(self, x_m, x_a):
        (z_m_p, z_m_s, z_a_p, z_a_s), (x_m_hat, x_a_hat), loss_code = self.d_net(x_m, x_a)

        logits_m = self.view_cls(z_m_s)
        logits_a = self.view_cls(z_a_s)


        return (z_m_p, z_m_s, z_a_p, z_a_s), (x_m_hat, x_a_hat), (logits_m, logits_a), loss_code


class KernelDecomposeNetwork(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input_m, n_input_a, n_z, code_num, data_num):
        super(KernelDecomposeNetwork, self).__init__()

        self.alpha = 1.0
        self.input_m_layer = nn.Linear(n_input_m, n_enc_1)
        self.input_a_layer = nn.Linear(n_input_a, n_enc_1)

        self.encoder_main = nn.Sequential(
                                    nn.Linear(n_enc_1, n_enc_2),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_2, n_enc_3),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_3, n_z),
                                )
        self.encoder_aux = nn.Sequential(
                                    nn.Linear(n_enc_1, n_enc_2),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_2, n_enc_3),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_3, n_z),
                                )
        self.encoder_shared_1 = nn.Sequential(
                                    nn.Linear(n_enc_1, n_enc_2),
                                    nn.ReLU(),
                                )
        
        self.encoder_shared_2 = nn.Sequential(
                                    nn.Linear(n_enc_2, n_enc_3),
                                    nn.ReLU(),
                                    nn.Linear(n_enc_3, n_z),
                                )
        

        self.sim_layer = BTNet([n_z, 128])

        self.kernel_layer_m = KernelLayer(data_num)
        self.kernel_layer_a = KernelLayer(data_num)

        self.mapping_layer = nn.Linear(data_num, n_z, bias=False)
        
        # pass continuous latent vector through discretization bottleneck
        # self.quantizer = VectorQuantizer(n_z, code_num, beta=0.25)
        
        self.decoder_m = nn.Sequential(
                                    nn.Linear(n_z, n_dec_1),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_1, n_dec_2),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_2, n_dec_3),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_3, n_input_m)
                                )
        
        self.decoder_a = nn.Sequential(
                                    nn.Linear(n_z, n_dec_1),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_1, n_dec_2),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_2, n_dec_3),
                                    nn.ReLU(),
                                    nn.Linear(n_dec_3, n_input_m)
                                )
        
        # self.kernel_centroid = nn.Parameter(torch.randn(code_num, data_num), requires_grad=True)

        high_feature_dim = 128
        n_view = 2

        self.specific_view = nn.Sequential(
            nn.Linear(n_z, high_feature_dim),
        )
        self.common_view = nn.Sequential(
            nn.Linear(n_z * n_view, high_feature_dim),
        )
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=n_z* n_view, nhead=1, dim_feedforward=256)

        self.prototype_net = PrototypeNet(n_z, 128, code_num)

        self.codebook = nn.Linear(n_z, code_num, bias=False)

        self.linear = nn.Sequential(
            nn.Linear(n_z, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, code_num),
            nn.Softmax(dim=1)
        )
        self.cluster_loss = ClusterLoss(code_num)
        self.ccd_loss = CGDDCLoss(code_num)

        self.contrastive_loss = Loss()


    def FAgg(self, x_m, x_a):
        enc1_m = F.relu(self.input_m_layer(x_m)) 
        enc1_a = F.relu(self.input_a_layer(x_a))
        
        z_m_p = self.encoder_main(enc1_m) # parity part
        z_a_p = self.encoder_aux(enc1_a)

        enc2_m = self.encoder_shared_1(enc1_m) # shared
        enc2_a = self.encoder_shared_1(enc1_a)

        z_m_s = self.encoder_shared_2(enc2_m)
        z_a_s = self.encoder_shared_2(enc2_a)

        z_s = torch.cat([z_m_s, z_a_s], dim=1)

        commonz, S = self.TransformerEncoderLayer(z_s)
        commonz = self.common_view(commonz) # normalize

        return commonz, S

    def get_qp(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.codebook.weight, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()  # Make sure each sample's n_values add up to 1. 

        p = target_distribution(q)
        return (q, p)

    def forward(self, x_m, x_a):

        enc1_m = F.relu(self.input_m_layer(x_m)) 
        enc1_a = F.relu(self.input_a_layer(x_a))
        
        z_m_p = self.encoder_main(enc1_m) # parity part
        z_a_p = self.encoder_aux(enc1_a)

        enc2_m = self.encoder_shared_1(enc1_m) # shared
        enc2_a = self.encoder_shared_1(enc1_a)


        z_m_s = self.encoder_shared_2(enc2_m)
        z_a_s = self.encoder_shared_2(enc2_a)


        _, z_dot_m = self.kernel_layer_m(z_m_s) 
        _, z_dot_a = self.kernel_layer_a(z_a_s)

        # _, z_dot= self.kernel_layer_m(z_s)

        z_mapping_m = self.mapping_layer(z_dot_m)
        z_mapping_a = self.mapping_layer(z_dot_a)

        # q_m, z_m_s_c, loss_m = self.quantizer(z_mapping_m) # the shared part
        # q_a, z_a_s_c, loss_a = self.quantizer(z_mapping_a)



        z_s = (z_m_s + z_a_s)
        
        z_m = z_m_p + z_s
        z_a = z_a_p + z_s

        loss_sim = self.sim_layer(z_m, z_a)

        x_m_hat = self.decoder_m(z_m)
        x_a_hat = self.decoder_a(z_a)


        # loss_proto, loss_vq = self.prototype_net(z_m_s, z_a_s)

        q_m, p_m = self.get_qp(z_m_s)
        q_a, p_a = self.get_qp(z_a_s)


        att_z_s, S = self.FAgg(x_m, x_a)



        q, p = self.get_qp(z_s)

 
        # loss_proto = F.kl_div(q_m.log(), p, reduction='batchmean') + F.kl_div(q_a.log(), p, reduction='batchmean')
        loss_proto = self.ccd_loss(q, z_s)

        soft_m = self.linear(z_m_s)
        soft_a = self.linear(z_a_s)
        soft_ = self.linear(z_s)

        
        att_z_s = F.normalize(att_z_s, dim=1)
        z_m_s_ = F.normalize(self.specific_view(z_m_s), dim=1)
        z_a_s_ = F.normalize(self.specific_view(z_a_s), dim=1)

        loss_cc = self.contrastive_loss(att_z_s, z_m_s_, S) + self.contrastive_loss(att_z_s, z_a_s_, S)

        return (z_m_p, z_m_s, z_a_p, z_a_s), (x_m_hat, x_a_hat), (loss_sim, loss_proto, loss_cc), q, (enc1_m, enc1_a, enc2_m, enc2_a)



class TransformerEncoderLayerWithLayerScale(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1, layer_scale_init_value=0.1):
        super(TransformerEncoderLayerWithLayerScale, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Layer Scale
        self.layer_scale1 = nn.Parameter(layer_scale_init_value * torch.ones((d_model,)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(layer_scale_init_value * torch.ones((d_model,)), requires_grad=True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 多头注意力机制
        attn_output, attn_output_weights = self.multihead_attn(src, src, src, attn_mask=src_mask, 
                                                               key_padding_mask=src_key_padding_mask)
        src = src + self.layer_scale1 * self.dropout1(attn_output)
        src = self.norm1(src)

        # 前馈网络
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.layer_scale2 * self.dropout2(ff_output)
        src = self.norm2(src)

        return src, attn_output_weights


class TrandformFusion(nn.Module):
    def __init__(self):
        super(TrandformFusion, self).__init__()
    
    def forward(self):
        pass