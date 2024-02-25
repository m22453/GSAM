from backbone import get_backbone
from models import *
from tqdm import tqdm
from sklearn.cluster import KMeans
import os, sys, json
import torch
import numpy as np
from torch.nn import functional as F
from processor_o import MiniNewsProcessor, NewsProcessor, ChineseNewsProcessor, BBCProcessor, ChineseNewsTripleProcessor, AgProcessor, AminerProcessor
from cm_plot import clustering_score, Logger
from mi_estimators import *
from sklearn.metrics import mutual_info_score
from torch.nn import MSELoss
from contrastive_loss import InstanceLoss, ClusterLoss, CrossViewConstraintsLoss

from torch.utils.data import DataLoader
import random
from datasets import MultiViewImgDataset, UCIDataset, S15Datasets
from ot import get_ot_loss, get_max_code
import warnings
warnings.filterwarnings('ignore')
import argparse
from utils import DataLogger, matching_acc, write_list_to_json
import datetime
from itertools import combinations


parser = argparse.ArgumentParser()
parser.description='please enter key parameters ...'


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parser.add_argument("-data", "--data", help="task name", type=str, default="uci", choices=['uci', 's15'])
parser.add_argument('-momentum', default=0.9, type=float, metavar='M',
                    help='momentum of Adam solver')
parser.add_argument('-wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',dest='weight_decay')
parser.add_argument('-align_view_type', '--align_view_type', default=0, type=int, choices=[0, 1]) # dest='type=0(tf-idf), type=1(bert)')
parser.add_argument('-batch_size', '--batch_size', default=128, type=int)
parser.add_argument('-seed', '--seed', default=100, type=int) # 100
parser.add_argument('-n_z', '--n_z', default=256, type=int)
parser.add_argument('-lr', '--lr', default=1e-4, type=float)
parser.add_argument('-total_epoch', '--total_epoch', default=151, type=int)
parser.add_argument('-start_mAlign_epoch', '--start_mAlign_epoch', default=61, type=int)
parser.add_argument('-start_fAlign_epoch', '--start_fAlign_epoch', default=91, type=int)
parser.add_argument('-log', '--log', default=0, type=int)
parser.add_argument('-dual_view', '--dual_view', default=0, type=int, choices=[0, 1]) # choose all or the dual
parser.add_argument('-mutual_align_layer', '--mutual_align_layer', default=-1, type=int, choices=[-1, 1, 2, 3])
parser.add_argument('-matching_scale', '--matching_scale', default=1, type=int)  # for matching numer devide by (matching_scale * num_labels)
parser.add_argument('-consolidation_mode', '--consolidation_mode', default='concat', type=str, choices=['max', 'mean', 'concat']) # 0: no consolidation, 1: only main view, 2: only aux view


view_trans_length = 5
args = parser.parse_args()

task_name = dataset = args.data

if args.log == 1:
    logger = DataLogger(task_name, './exps/main_results.log')
    sys.stdout = logger

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
seed = args.seed
# seed init.
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# torch seed init.
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


data_dir = './data/' + dataset
print('==*== ' * 15)
print('Task name = ', task_name)
print('Parameters = ', args)

if 'chinese_news' not in task_name:
    GLOBAL_LENGTH = None


num_views_task = {
    "mini_news": 2,
    "news": 2,
    "chinese_news": 2,
    "bbc": 2,
    "chinese_news_3": 3,
    "ag_news": 2,
    "aminer" : 4,
    "s15": 2,
    "uci": 3
}

num_labels_task = {
    "mini_news": 3,
    "news": 10,  # 41,
    "chinese_news": 4,
    "bbc": 5,
    "chinese_news_3": 4,
    "ag_news": 4,
    "aminer": 3,
    "s15": 15,
    "uci": 10
}

num_labels = num_labels_task[task_name]
num_views = num_views_task[task_name]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
if torch.cuda.is_available():
    print('cuda device = ', torch.cuda.current_device())


if task_name == "uci":
    dataset = UCIDataset()
elif task_name == "s15":
    dataset = S15Datasets()


x_num = len(dataset)
all_data_loader = DataLoader(dataset, batch_size=x_num, shuffle=False)
print(dataset.x_n)
print(dataset.y.shape)
labels = dataset.y

for (v_lst, _) in tqdm(all_data_loader):
    embeddings_list = []
    for i in range(num_views):
        embeddings_list.append(v_lst[i].float().to(device))

if num_views == 2:
    embeddings_m, embeddings_a = embeddings_list[0], embeddings_list[1]
else: # TODO: for dealing with the multi-view case
    embeddings_m, embeddings_a = embeddings_list[:1], embeddings_list[1:]

    # TODO: for dealing with the multi-view case
    if type(embeddings_m) == list:

        # if args.consolidation_mode == 'max':
        #     embeddings_m = np.maximum.reduce(embeddings_m)
        # elif args.consolidation_mode == 'mean':
        #     embeddings_m = np.sum(embeddings_m, axis=0) / len(embeddings_m) * 1.0
        # elif args.consolidation_mode == 'concat':
        #     embeddings_m = np.concatenate(embeddings_m, axis=1)
        embeddings_m = torch.cat(embeddings_m, dim=1)

        print(f"embeddings_m.shape = {embeddings_m.shape}")
    
    if type(embeddings_a) == list:

        embeddings_a = torch.cat(embeddings_a, dim=1)
        print(f"embeddings_a.shape = {embeddings_a.shape}")


embeddings_list = [embeddings_m, embeddings_a]

for e in embeddings_list:
    print(type(e))

print(f"embeddings_m.shape = {embeddings_m.shape} &  embeddings_a.shape = {embeddings_a.shape}")

km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(embeddings_m.detach().cpu().numpy())
predict_labels = km.labels_
print(f'embeddings_m clustering score = {clustering_score(labels, predict_labels)}')

km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(embeddings_a.detach().cpu().numpy())
predict_labels = km.labels_
print(f'embeddings_a clustering score = {clustering_score(labels, predict_labels)}')

# for training

x_num = embeddings_m.shape[0]
x_dim_m = embeddings_m.shape[1]
x_dim_a = embeddings_a.shape[1]


model = CrossModal(
        n_input_m=x_dim_m,
        n_input_a=x_dim_a,
        n_z=args.n_z,
        code_num=num_labels).to(device)


dataset = MultiViewImgDataset(embeddings_list, labels)

all_data_loader = DataLoader(dataset, batch_size=x_num, shuffle=False)

# batchembeddings_a.shape[0]

for (v_lst, _) in all_data_loader:

    embeddings_m = v_lst[0].float().to(device)
    embeddings_a = v_lst[1].float().to(device)

    with torch.no_grad():
        (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat, q_m, q_a, _, _ = model(embeddings_m, embeddings_a)

# assert len(z1_m) == x_num
# model.summary()


z = (z1_m + z2_m + z1_a + z2_a) / 4.0
z = z.detach().cpu().numpy()
# np.savetxt(f"./training/{task_name}_notrain.txt", z)
# np.savetxt(f"./training/{task_name}_labels.txt", labels)
km = KMeans(n_clusters= num_labels, random_state=seed, init='k-means++').fit(z)
predict_labels = km.labels_
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
res = clustering_score(labels, predict_labels)
print(f"init clustering score = {res}")

print(f"the shape of km.cluster_centers_ is {km.cluster_centers_.shape}")
# print(f"the shape of codebook is {model.quantizer.codebook.weight.shape}")
# model.quantizer.codebook.weight.data = torch.tensor(km.cluster_centers_).to(device)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


# parameters_lst = list(model.parameters())

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # for all parameters except min_mi and ot parameters


mse = MSELoss(reduction = 'mean') # for reconstruction loss
cluster_loss = ClusterLoss(args.n_z).to(device) # for contrastive loss 

batch_size = args.batch_size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # training for shuffle
    
best = 0.0
best_res = {}

res_list = []

for i in range(args.total_epoch):
    
    res_dict = {}
    # zs = []
    model.train()
    
    for j, (xs, _) in enumerate(data_loader):
        # print(f"v_lst[0].shape = {v_lst[0].shape} \nv_lst[1].shape = {v_lst[1].shape} \nlabels.shape = {labels}")

        embeddings_m = xs[0].float().to(device)
        embeddings_a = xs[1].float().to(device)

        # print(f"xs[0].shape = {xs[0].shape}")

        if i < args.start_mAlign_epoch:
            (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat, q_m, q_a, _, loss_bt = model(embeddings_m, embeddings_a, type='warm_up')
        # (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat = model(embeddings_m, embeddings_a)
        elif i < args.start_fAlign_epoch: # 141
            (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat, q_m, q_a, loss_code, loss_bt = model(embeddings_m, embeddings_a, type='mutual_align')
        else:
            # 2e-4
            # if args.data == 's15':
            #     lr = 2e-3
            # else:
            lr = args.lr
            optimizer = torch.optim.Adam(model.parameters(), lr=lr) # for all parameters except min_mi and ot parameters
            (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat, q_m, q_a, loss_code, loss_bt = model(embeddings_m, embeddings_a, type='fusion_align')

            

        # loss_recon = mse(x_main_hat, embeddings_m) + mse(x_aux_hat, embeddings_a) 
        # loss_recon = F.mse_loss(x_hat, embeddings_m) #+ F.mse_loss(embeddings_a, x_aux_hat)
        loss_cluster = cluster_loss(z2_m, z2_a)

        loss = loss_bt
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        res_dict['loss'] = loss.item()


    # print('epoch = {}, loss = {}'.format(i, loss.item()))
    if i % 10 == 0:
        # print('model evaling ...')
        model.eval()
        test_loader = DataLoader(dataset, batch_size=x_num, shuffle=False)
        for (m, a), ls in test_loader:

            # print(f"original labels shape = {ls.shape}")
            embeddings_m = m.float().to(device)
            embeddings_a = a.float().to(device)

            with torch.no_grad():
                (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat, q_m, q_a, fagg, _ = model(embeddings_m, embeddings_a)
                # (z1_m, z2_m, z1_a, z2_a), _, _ = model(embeddings_m, embeddings_a)
            ls = ls.view(x_num)


        assert len(z2_m) == x_num
        
        def km_test(z, ls): # input the embeddings and labels
            z = (z).detach().cpu().numpy()
            ls = ls.detach().cpu().numpy()
            km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
            predict_labels = km.labels_
            res = clustering_score(ls, predict_labels)
            return res
        
        res1 = km_test(z1_m, ls)
        res2 = km_test(z1_a, ls)
        res3 = km_test(z2_m, ls)
        res4 = km_test(z2_a, ls)
        res5 = km_test(fagg, ls)
        print(f'epoch --> {i}, km_test(z1_m, ls) = {res1}')
        print(f'epoch --> {i}, km_test(z1_a, ls) = {res2}')
        print(f'epoch --> {i}, km_test(z2_m, ls) = {res3}')
        print(f'epoch --> {i}, km_test(z2_a, ls) = {res4}')
        print(f'epoch --> {i}, km_test(FAgg, ls) = {res5}')

        res_dict['z2_m'] = res3
        res_dict['z2_a'] = res4
        res_dict['fagg'] = res5


        if i < args.start_fAlign_epoch:
            # for init the codebook weight
            z = (z2_m + z2_a) / 2.0
            z = z.detach().cpu().numpy()
            km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
            model.codebook.weight.data = torch.tensor(km.cluster_centers_).to(device)
            # TODO: use the normalized codebook
        else:
            pass

        # if args.log:
        #     if i == args.start_mAlign_epoch:
        #         z = (z2_m + z2_a) / 2.0
        #         z = z.detach().cpu().numpy()
        #         np.savetxt(f"./training/{task_name}_o_add.txt", z)

        #         z = torch.cat((z2_m, z2_a), dim=1)
        #         z = z.detach().cpu().numpy()
        #         np.savetxt(f"./training/{task_name}_o_cat.txt", z)
            
        #     if i == args.start_fAlign_epoch:
        #         z = (z2_m + z2_a) / 2.0
        #         z = z.detach().cpu().numpy()
        #         np.savetxt(f"./training/{task_name}_mu_add.txt", z)

        #         z = torch.cat((z2_m, z2_a), dim=1)
        #         z = z.detach().cpu().numpy()
        #         np.savetxt(f"./training/{task_name}_mu_cat.txt", z)

        #     if i == args.total_epoch - 1:
        #         z = (z2_m + z2_a) / 2.0
        #         z = z.detach().cpu().numpy()
        #         np.savetxt(f"./training/{task_name}_fa_add.txt", z)

        #         z = torch.cat((z2_m, z2_a), dim=1)
        #         z = z.detach().cpu().numpy()
        #         np.savetxt(f"./training/{task_name}_fa_cat.txt", z)

        z = torch.add(z2_m, z2_a)
        z = z.detach().cpu().numpy()
        km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
        pred_y = km.labels_
        res = clustering_score(ls.detach().cpu().numpy(), pred_y)
        print(f"fusion res by adding = {res}")

        res_dict['adding'] = res

        z = torch.cat((z2_m, z2_a), dim=1)
        z = z.detach().cpu().numpy()
        km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
        pred_y = km.labels_
        res = clustering_score(ls.detach().cpu().numpy(), pred_y)
        print(f"fusion res by catting = {res}")

        res_dict['catting'] = res

        
        # if i % 40 == 0:
        #     k = int(len(ls.detach().cpu().numpy()) / num_labels / args.matching_scale)
        #     m2a_acc, a2m_acc = matching_acc(z2_m, z2_a, k), matching_acc(z2_a, z2_m, k)
        #     print(f"matching_k = {k}, m2a_acc = {m2a_acc}, a2m_acc = {a2m_acc}")

        #     res_dict['m2a_acc'] = m2a_acc
        #     res_dict['a2m_acc'] = a2m_acc

        # print(res)
        if res['NMI'] > best and i >= args.start_mAlign_epoch:
            best = res['NMI']
            best_res = res
            # torch.save(encoder.state_dict(), 'encoder.pth')
            # torch.save(decoder.state_dict(), 'decoder.pth')

        print('\n')

        res_list.append(res_dict)

# if args.log == 1:
#     write_list_to_json(res_list, f'{task_name}_test_res_list.json')

print(f'-- {task_name}----- final res via k-means =  {best_res}--------')
