import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from random import choice
from cm_plot import view_weights



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InstanceLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        
        batch_size = len(z_i)
        mask = self.mask_correlated_samples(batch_size)
        

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        # sim = torch.exp(sim) # optional
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
    
class WeightedInstanceLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(WeightedInstanceLoss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        
        batch_size = len(z_i)
        mask = self.mask_correlated_samples(batch_size)
        
        # get the weights
        w_vec = view_weights(z_i, z_j, 30).to(device)

        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        # sim = torch.exp(sim) # optional
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) * w_vec.repeat(2, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

def contrastive_triple_bySim(x, y):
    """
    x: anchor matrix
    y: positive matrix
    """
    def get_sim(x):
        x = x / torch.norm(x, dim=1, keepdim=True)
        sim = torch.matmul(x, x.T)
        return sim

    # three matrix
    triple_list = [[],[],[]]
    for i, a in enumerate(x):

        p = y[i]
        sim = get_sim(x)
        _, top_k_indices = torch.topk(sim[i], k=3, largest=False)
        # print(top_k_indices[1:])
        n_i = choice(top_k_indices)
        n = y[n_i]

        # append to trget list
        triple_list[0].append(a)
        triple_list[1].append(p)
        triple_list[2].append(n)

    centroid_margin_loss = torch.nn.TripletMarginLoss()
    loss = centroid_margin_loss(torch.stack(triple_list[0]), torch.stack(triple_list[1]),torch.stack(triple_list[2]))

    return loss

class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss \n")


    def forward(self, features_1, features_2, negative_mask=None, margin=0.0):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        if negative_mask is not None:
            mask = mask & negative_mask
        
        pos = torch.exp(torch.sum(features_1*features_2 - margin, dim=-1) / self.temperature)
        print(f'pos.shape = {pos.shape}')
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        print(f'neg.shape = {neg.shape}')
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        print(f'mask = {mask}')
        
        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)
            
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        
        return {"loss":loss_pos, "pos_mean":pos_n.detach().cpu().numpy(), "neg_mean":neg_mean.detach().cpu().numpy(), "pos":pos.detach().cpu().numpy(), "neg":neg.detach().cpu().numpy()}

class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits 锚点样本和所有特征的相似度
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask), # 创建一个与 mask 形状相同的全为 1 的张量
                1, # 沿着 dim=1 的方向进行 scatter
                torch.arange(batch_size).view(-1, 1).to(device),  # 创建一个列向量，包含从 0 到 batch_size-1 的索引
                0 #指定要用来填充的值。在这里，将索引位置填充为 0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature=1.0):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class Loss(nn.Module):
    def __init__(self, temperature_f=0.05):
        super(Loss, self).__init__()
        self.temperature_f = temperature_f
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, h_i, h_j, S):
        

        max_vals, _ = torch.max(S, dim=1, keepdim=True)
        min_vals, _ = torch.min(S, dim=1, keepdim=True)

        epsilon = 1e-8
        normalized_output = (S - min_vals) / (max_vals - min_vals + epsilon)

        # S = normalized_output

        batch_size = h_i.size(0)
        self.mask = self.mask_correlated_samples(batch_size)

        S_1 = S.repeat(2, 2)
        all_one = torch.ones(batch_size*2, batch_size*2).to('cuda')
        S_2 = all_one - S_1
        N = 2 * batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim1 = torch.multiply(sim, S_2)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss





# x = torch.randn(4, 6)
# y = torch.randn(4, 6)

# c = torch.cat([x, y], dim=0)
# c = c.div(c.norm(p=2, dim=1, keepdim=True) + 1e-12)

# cosine = torch.mm(c, c.t())
# print(f'cosine = {cosine < 0.7}')

# pc_loss = PairConLoss()
# loss = pc_loss(x, y)


# initial_delta = 0.001
# growth_factor = 1.002
# update_frequency = 1000

# def compute_delta(step):
#     growth_count = step // update_frequency
    
#     growth_multiplier = growth_factor ** growth_count
    
#     current_delta = initial_delta * growth_multiplier
    
#     return current_delta

# for step in range(5000):
#     delta = compute_delta(step)
#     if step % 1000 == 0:
#         print(f"Step {step + 1}: Delta = {delta}")


# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# # 设置样本数量和维度
# num_samples = 10
# embedding_dim = 128

# # 生成随机样本矩阵
# samples = np.random.rand(num_samples, embedding_dim)

# # 设置每个样本产生的近邻数量
# k_neighbors = 5

# # 计算每对样本近邻的交集
# intersection_matrix = np.zeros((num_samples, num_samples), dtype=int)

# # 计算余弦相似度矩阵
# cosine_sim_matrix = cosine_similarity(samples)
# print(cosine_sim_matrix)



# for i in range(num_samples):
#     for j in range(num_samples):
#         # 找到近邻索引
#         neighbors_i = np.argsort(cosine_sim_matrix[i])[-k_neighbors:]
#         neighbors_j = np.argsort(cosine_sim_matrix[j])[-k_neighbors:]

#         # 计算交集大小
#         intersection = np.intersect1d(neighbors_i, neighbors_j)

#         # 将交集大小填入矩阵
#         intersection_matrix[i, j] = len(intersection)


# # Print the intersection matrix
# print("Intersection Matrix:")
# print(intersection_matrix)



# import torch
# import torch.nn.functional as F

# # Set the number of samples and dimensions
# num_samples = 10
# embedding_dim = 128

# # Generate random sample tensor
# samples = torch.from_numpy(samples).float()

# # Set the number of neighbors to find
# k_neighbors = 5

# # Initialize the intersection matrix
# intersection_matrix = torch.zeros((num_samples, num_samples), dtype=torch.int)

# # Find neighbors and calculate intersection
# for i in range(num_samples):
#     for j in range(num_samples):
#         # Calculate cosine similarity
#         cosine_sim = F.cosine_similarity(samples[i].unsqueeze(0), samples, dim=1)

#         # Find neighbors indices
#         neighbors_i = torch.argsort(cosine_sim, descending=True)[:k_neighbors]

#         cosine_sim = F.cosine_similarity(samples[j].unsqueeze(0), samples, dim=1)
#         neighbors_j = torch.argsort(cosine_sim, descending=True)[:k_neighbors]

#         # Calculate intersection size
#         intersection = torch.tensor(list(set(neighbors_i.numpy()) & set(neighbors_j.numpy())))


#         # Fill the intersection matrix
#         intersection_matrix[i, j] = len(intersection)

# # Print the intersection matrix
# print("Intersection Matrix:")
# print(intersection_matrix)


class CrossViewConstraintsLoss(nn.Module):
    def __init__(self, thr) -> None:
        super(CrossViewConstraintsLoss, self).__init__()
        """
        thr in [1/K, 1]
        weighted in {True, False}
        """
        self.thr = thr

    def build_constraints_graph_by_similarity(self, Xs):
        """
        Xs is the feature matrix of multi-view data
        """
        graphs = []
        for X in Xs:
            output_norm = F.normalize(X, p=2, dim=1)
            sim_mat = torch.matmul(output_norm, output_norm.t())
            sim_mat[sim_mat > self.thr] = 1
            sim_mat[sim_mat <= self.thr] = 0
            graphs.append(sim_mat)
        return graphs
    
    def triple_margin_loss(self, A, X):
        triple_lst, triple_set = [[], [], []], set()
        for i, x in enumerate(X):
            p_mask = torch.where(A[i]>0)[0]
            n_mask = torch.where(A[i] == 0)[0]
            for p in p_mask.detach().cpu().numpy().tolist():
                n = choice(n_mask.detach().cpu().numpy().tolist())
                t_pair = (i, p, n)
                if t_pair not in triple_set:
                    triple_set.add(t_pair)

                    triple_lst[0].append(x) #anchor
                    triple_lst[1].append(X[p]) #positive
                    triple_lst[2].append(X[n]) #negative
                else:
                    continue
        triple_margin_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        anchor = torch.stack(triple_lst[0])
        positive = torch.stack(triple_lst[1])
        negative = torch.stack(triple_lst[2])
        loss = triple_margin_loss(anchor, positive, negative)
        return loss
    
    def forward(self, Xs):
        self.graphs = self.build_constraints_graph_by_similarity(Xs=Xs)
        loss = 0 
        for i, X in enumerate(Xs):
            X_nor = F.normalize(X, p=2, dim=1) # for the balance of feature scale
            A = self.graphs[i]
            for j, g in enumerate(self.graphs):
                if i == j:
                    continue
                else:
                    A += g
            A[A >= 1] = 1
            # except diag
            diag = torch.eye(A.size(0)).to(device)
            A = A - diag

            loss = loss + self.triple_margin_loss(A, X_nor)
        return loss

