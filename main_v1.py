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
from contrastive_loss import InstanceLoss, ClusterLoss, CrossViewConstraintsLoss

from torch.utils.data import DataLoader
import random
from datasets import MultiViewDataset
import warnings
warnings.filterwarnings('ignore')
import argparse
parser = argparse.ArgumentParser()
parser.description='please enter key parameters ...'



os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parser.add_argument("-data", "--data", help="task name", type=str, default="bbc")
parser.add_argument('-momentum', default=0.9, type=float, metavar='M',
                    help='momentum of Adam solver')
parser.add_argument('-wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',dest='weight_decay')
parser.add_argument('-align_view_type', '--align_view_type', default=1, type=int, choices=[0, 1]) # dest='type=0(tf-idf), type=1(bert)')
parser.add_argument('-batch_size', '--batch_size', default=256, type=int)
parser.add_argument('-seed', '--seed', default=42, type=int)
parser.add_argument('-n_z', '--n_z', default=128, type=int)
parser.add_argument('-lr', '--lr', default=1e-4, type=float) 
parser.add_argument('-freeze_prototypes_epochs', '--freeze_prototypes_epochs', default=3, type=int)
parser.add_argument('-warm_up_epochs', '--warm_up_epochs', default=40, type=int)

view_trans_length = 5
args = parser.parse_args()

task_name = dataset = args.data

if task_name == 'ag_news':  # num_train_epochs = 8
    dataset = 'ag_news/test'
elif task_name == 'news':  # num_train_epochs = 6
    dataset = 'news/News_Category_Dataset_v2'
elif task_name == 'bbc':  # num_train_epochs = 10
    dataset = 'bbc'
elif task_name == 'aminer':
    dataset = 'aminer/AMiner_mv_4v'
elif task_name == 'mini_news':  # num_train_epochs = 12
    dataset = 'news/News_Category_Dataset_mini'
elif task_name == 'chinese_news' or task_name == 'chinese_news_3':  # num_train_epochs = 6
    dataset = 'chinese_news/fileOfTT'


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
print('Dataset dir = ', data_dir)
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
    "aminer" : 4
}

num_labels_task = {
    "mini_news": 3,
    "news": 10,  # 41,
    "chinese_news": 4,
    "bbc": 5,
    "chinese_news_3": 4,
    "ag_news": 4,
    "aminer": 3
}

num_labels = num_labels_task[task_name]
num_views = num_views_task[task_name]


processors = {
    "mini_news": MiniNewsProcessor,
    "news": NewsProcessor,
    "chinese_news": ChineseNewsProcessor,
    "bbc": BBCProcessor,
    "chinese_news_3": ChineseNewsTripleProcessor,
    "ag_news": AgProcessor,
    "aminer": AminerProcessor
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
if torch.cuda.is_available():
    print('cuda device = ', torch.cuda.current_device())

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

if task_name in ['aminer', 'chinese_news_3', 'chinese_news']:
    processor = processors[task_name](view_trans_length)
    print('GLOBAL_LENGTH = {}'.format(view_trans_length))
else:
    processor = processors[task_name]()

data_list = processor.get_train_examples(data_dir) # corresponding data 

keys = list(data_list[0].keys())
print('keys:', keys)
v_lst = [[item[k] for item in data_list] for k in keys if k!='label']
labels = np.array([item['label'] for item in data_list]) # y_true
print(f"original labels shape = {labels.shape}")

backbone_m = get_backbone(1)
backbone_a = get_backbone(args.align_view_type)

# TODO: add the selection of view
embeddings_m = backbone_m(v_lst[1]) # main view embedding
embeddings_a = backbone_a(v_lst[0])

embeddings_list = [embeddings_m, embeddings_a]  # for model training

# import scipy.io as sio
# sio.savemat('embeddings_bbc.mat', {'embeddings_m': embeddings_m, 'labels': labels, 'embeddings_a': embeddings_a})


print(f"embeddings_m.shape = {embeddings_m.shape} \nembeddings_a.shape = {embeddings_a.shape}")

km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(embeddings_m)
predict_labels = km.labels_
print(f'embeddings_m clustering score = {clustering_score(labels, predict_labels)}')

km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(embeddings_a)
predict_labels = km.labels_
print(f'embeddings_a clustering score = {clustering_score(labels, predict_labels)}')

# for training

x_num = embeddings_m.shape[0]
x_dim_m = embeddings_m.shape[1]
x_dim_a = embeddings_a.shape[1]


# model = CrossViewModel(
#             n_enc_1=600,
#             n_enc_2=200,
#             n_enc_3=2000,
#             n_dec_1=2000,
#             n_dec_2=200,
#             n_dec_3=600,
#             n_input_m=x_dim_m,
#             n_input_a=x_dim_a,
#             n_z=args.n_z,
#             code_num=num_labels,

#         ).to(device)

# bs = args.batch_size
bs = x_num

model = KernelDecomposeNetwork(
            n_enc_1=600,    
            n_enc_2=200,
            n_enc_3=2000,
            n_dec_1=2000,
            n_dec_2=200,
            n_dec_3=600,
            n_input_m=x_dim_m,
            n_input_a=x_dim_a,
            n_z=args.n_z,
            code_num=num_labels,
            data_num=bs
        ).to(device)


dataset = MultiViewDataset(embeddings_list, labels)
# all_data_loader = DataLoader(dataset, batch_size=x_num, shuffle=False)


# for (v_lst, _) in all_data_loader:

#     embeddings_m = v_lst[0].to(device)
#     embeddings_a = v_lst[1].to(device)

#     with torch.no_grad():
#         (z1_m, z2_m, z1_a, z2_a), x_main_hat, x_aux_hat, q_m, q_a, _, _ = dNet(embeddings_m, embeddings_a)


# z = (z1_m + z2_m + z1_a + z2_a) / 4.0
# z = z.detach().cpu().numpy()
# km = KMeans(n_clusters= num_labels, random_state=seed, init='k-means++').fit(z)
# predict_labels = km.labels_
# print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
# res = clustering_score(labels, predict_labels)
# print(f"init clustering score = {res}")

# print(f"the shape of km.cluster_centers_ is {km.cluster_centers_.shape}")
# print(f"the shape of codebook is {model.quantizer.codebook.weight.shape}")
# model.quantizer.codebook.weight.data = torch.tensor(km.cluster_centers_).to(device)

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


# parameters_lst = list(model.parameters())

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # for all parameters except min_mi and ot parameters
mse = torch.nn.MSELoss() # for reconstruction loss
bce = torch.nn.BCELoss() # for binary cross entropy loss
cce = torch.nn.CrossEntropyLoss() # for cross entropy loss

batch_size = args.batch_size
data_loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    
best = 0.0
best_res = {'warm_up': {}, 'final': {}}

def normalize(x):
    return F.normalize(x, p=2, dim=1)

from task_loss import CGDDCLoss
from kernel_opts import cosine, gaussian, polynomial, laplacian, multi_kernel
ddc_loss = CGDDCLoss(num_labels, use_l2_flipped=False)
for i in range(100):

    model.train()

    for j, (xs, _) in enumerate(data_loader):

        x_m = xs[0].to(device)
        x_a = xs[1].to(device)

        # feed forward
        (z_m_p, z_m_s, z_a_p, z_a_s), (x_m_hat, x_a_hat), losses, q, (enc1_m, enc1_a, enc2_m, enc2_a) = model(x_m, x_a)


        loss_sim, loss_proto, loss_cc = losses

        loss_recon = mse(x_m_hat, x_m) + mse(x_a_hat, x_a)
        
        tmp_m, tmp_a = torch.mm(normalize(z_m_s), normalize(z_m_p.t())), torch.mm(normalize(z_a_s), normalize(z_a_p.t()))
        loss_differ = torch.div(torch.norm(tmp_m) + torch.norm(tmp_a), z_m_s.shape[0])

        tmp_m, tmp_a = torch.mm(normalize(z_m_s), normalize(z_m_s.t())), torch.mm(normalize(z_a_s), normalize(z_a_s.t()))
        tmp_m_k, tmp_a_k = polynomial(tmp_m), polynomial(tmp_a)

        z_s = torch.cat([z_m_s, z_a_s], dim=1)
        tmp_s = torch.mm(normalize(z_s), normalize(z_s.t()))
        tmp_s = polynomial(tmp_s)
        
        loss_dot = cce(tmp_m_k, tmp_a_k) + cce(tmp_a_k, tmp_m_k) + cce(tmp_m_k, tmp_s) + cce(tmp_a_k, tmp_s) 
        # + cce(kernel_opt(enc1_m), kernel_opt(enc1_a)) + cce(kernel_opt(enc2_m), kernel_opt(enc2_a))

        # if i == 0:
        #     # init shared clustering certriods
        #     z_s = torch.cat([z_m_s, z_a_s], dim=1)
        #     tmp = torch.mm(normalize(z_s), normalize(z_s.t()))
        #     tmp = cosine(tmp)
        #     z = (tmp).detach().cpu().numpy()
        #     km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
        #     model.kernel_centroid.data = torch.tensor(km.cluster_centers_).to(device)
        #     print(f"zz.shape = {z.shape}, model.kernel_centroid.shape = {model.kernel_centroid.shape}")


        # def get_q(zz):
        #     q = 1.0 / (1.0 + torch.sum(torch.pow(zz.unsqueeze(1) - model.kernel_centroid, 2), 2) / 1.0)
        #     q = q.pow((1.0 + 1.0) / 2.0)
        #     q = (q.t() / torch.sum(q, 1)).t()  # Make sure each sample's n_values add up to 1.
        #     return q

        # loss_task = ddc_loss(get_q(tmp_m_k), z_m_s) + ddc_loss(get_q(tmp_a_k), z_a_s)

        '''
        logits = torch.cat([logits_m, logits_a], dim=0)
        targets = torch.cat([torch.ones_like(logits_m), torch.zeros_like(logits_a)], dim=0)
        loss_dann = bce(logits, targets)
        

        loss_consistency = torch.div(
            torch.norm(torch.mm(normalize(z_m_s.t()), normalize(z_m_s)) - torch.mm(normalize(z_a_s.t()), normalize(z_a_s))),
            batch_size
        )
        '''

        
        
        # print(f"epoch = {i}, iter = {j}----,  \
        #     \nloss_recon = {loss_recon.item()}, \
        #     \nloss_differ = {loss_differ.item()},   \
        #     \nloss_dann = {loss_dann.item()}, \
        #     \nloss_mmd = {loss_mmd.item()}\n-------")

        loss_dic = {
            "loss_recon": loss_recon,
            "loss_differ": loss_differ * 1,
            "loss_sim": loss_sim * 1,
            "loss_dot": loss_dot * 0.005,
            "loss_proto": loss_proto * 10,
            "loss_cc": loss_cc * 1,
        }
        if i < args.warm_up_epochs:
            loss = loss_recon * 1 + 1 * loss_differ + 1 * loss_sim 
        elif i < 50:
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-3) # for all parameters except min_mi and ot parameters
            loss = loss_cc + loss_dot * 0.00001
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) # for all parameters except min_mi and ot parameters
            loss = loss_proto * 0.001 + loss_recon 
        print(f'loss_dic = {loss_dic}')


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        

    if i % 1 == 0:
        # print('model evaling ...')
        model.eval()
        test_loader = DataLoader(dataset, batch_size=x_num, shuffle=False)
        for (m, a), ls in test_loader:

            # print(f"original labels shape = {ls.shape}")
            x_m = m.to(device)
            x_a = a.to(device)

            with torch.no_grad():
                (z_m_p, z_mapping_m, z_a_p, z_mapping_a), (x_m_hat, x_a_hat), _, q, _ = model(x_m, x_a)
            
            ls = ls.view(x_num)

        q = q.detach().cpu().numpy()
        y_q = q.argmax(axis=1)
        res = clustering_score(ls.detach().cpu().numpy(), y_q)
        print(f"""epoch = {i}, q.argmax res = {res}""")


        # y_m = q_m.detach().cpu().numpy().argmax(axis=1)
        # y_a = q_a.detach().cpu().numpy().argmax(axis=1)

        # print(clustering_score(ls.detach().cpu().numpy(), y_m), clustering_score(ls.detach().cpu().numpy(), y_a))
        
        z = (z_mapping_m).detach().cpu().numpy()
        km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
        predict_labels = km.labels_
        res = clustering_score(ls.detach().cpu().numpy(), predict_labels)
        print(f"""epoch = {i}, res = {res}""")

        z = (z_mapping_a).detach().cpu().numpy()
        km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
        predict_labels = km.labels_
        res = clustering_score(ls.detach().cpu().numpy(), predict_labels)
        print(f"""epoch = {i}, res = {res}""")

        z = (torch.cat([z_mapping_m, z_mapping_a], dim=1)).detach().cpu().numpy()
        km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
        predict_labels = km.labels_
        res = clustering_score(ls.detach().cpu().numpy(), predict_labels)
        print(f"""epoch = {i}, cating fusion res = {res}""")

        # if res['NMI'] > best:
        #     best = res['NMI']
        #     if i <= args.warm_up_epochs:
        #         best_res['warm_up'] = res
        #     else:
        #         best_res['final'] = res

        z = (torch.add(z_mapping_m, z_mapping_a)).detach().cpu().numpy()
        km = KMeans(n_clusters=num_labels, random_state=seed, init='k-means++').fit(z)
        predict_labels = km.labels_
        res = clustering_score(ls.detach().cpu().numpy(), predict_labels)
        print(f"""epoch = {i}, adding fusion res = {res}""")
        if i < args.warm_up_epochs:
            model.codebook.weight.data = torch.tensor(km.cluster_centers_).to(device)
            model.prototype_net.prototype_layer.weight.data = torch.tensor(km.cluster_centers_).to(device)

        print('-------------------------------------------')

        if res['NMI'] > best:
            best = res['NMI']
            if i <= args.warm_up_epochs:
                best_res['warm_up'] = res
            else:
                best_res['final'] = res
            # torch.save(encoder.state_dict(), 'encoder.pth')
            # torch.save(decoder.state_dict(), 'decoder.pth')

print(f'------- final res =  {best_res}--------')


# def load_encoder(encoder_path='encoder.pth', embeddings=embeddings, labels=labels):
#     encoder = Encoder(x_dim, 512, 128).to(device)
#     encoder.load_state_dict(torch.load(encoder_path))
#     embeddings = torch.tensor(embeddings, dtype=torch.float).to(device)
#     z = encoder(embeddings)
#     z = z.detach().cpu().numpy()
#     km = KMeans(n_clusters=num_labels, random_state=44, init='k-means++').fit(z)
#     predict_labels = km.labels_
#     res = clustering_score(labels, predict_labels)
#     print(res)
#     return encoder

# # load_encoder(encoder_path='encoder.pth', embeddings=embeddings, labels=labels)