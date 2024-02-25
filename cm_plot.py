# -*- coding: utf-8 -*-
# @Time : 2021/8/31 15:18
# @Author : ruinabai_TEXTCCI
# @FileName: utils.py
# @Email : m15661362714@163.com
# @Software: PyCharm

# @Blog ：https://www.jianshu.com/u/3a5783818e3a


import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import torch
from sklearn import metrics
from munkres import Munkres


import sys  # 需要引入的包


# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="record_log.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def view_weights(x, y, k):
    """
    x: (n_samples, n_features)
    y: (n_samples, n_features)
    k: int, number of knn

    return view weights (n_samples, 1)
    """
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    n_samples, n_features = x.shape
    assert x.shape == y.shape
    assert n_samples > k

    # x = x / np.linalg.norm(x, axis=1, keepdims=True)
    # y = y / np.linalg.norm(y, axis=1, keepdims=True)

    nbrs_x = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(x)
    _, indices_x = nbrs_x.kneighbors(x)
    indices_x = indices_x[:, 1:]

    nbrs_y = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(y)
    _, indices_y = nbrs_y.kneighbors(y)
    indices_y = indices_y[:, 1:]

    # &
    common_nbrs = np.array([len(set(indices_x[i]) & set(indices_y[i])) for i in range(n_samples)])

    weights = common_nbrs / np.max(common_nbrs)

    weights = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

    return weights
    


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def cluster_acc(y_true, y_pred):

    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    acc_dec, f1_dec  = cluster_acc(y_true, y_pred)
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred), 4),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred), 4),
            'ARI': round(adjusted_rand_score(y_true, y_pred), 4),
            'ACC_DEC': round(acc_dec, 4),
            'F1_DEC': round(f1_dec, 4)}


def plot_confusion_matrix(cm, classes, title='', normalize=False,
                           figsize=(12, 10),
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)



    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('./cms/'+title + ".png", format='png', dpi=150)



# cm = np.array([
# [1339, 294, 40, 28, 397, 14, 13, 16, 24, 13],
# [ 26, 716, 49  , 46 , 102  , 52  , 16  , 25 , 76, 21],
# [999, 155, 27, 14, 2, 48, 11, 7, 48, 12],
# [  14 ,  98 ,  16 , 684 , 110 ,  45 ,   7 ,   7 ,  15 ,   8],
# [   6 , 120 ,  53 , 632  ,213   ,33  ,  6  , 19  , 44,   18],
# [  17  , 98 ,  53 ,  61 ,   1,1343 ,   6 ,  45  , 64  , 19],
# [  16 , 225 ,  32 ,  29 , 689 ,   4, 1475,   14,   64 ,   8],
# [  12 , 182 ,  57 ,  16 ,   0  , 42 ,  23 ,3054 ,  27 ,  13],
# [  28  ,108 ,  64 ,  26 , 716 ,  15  , 33,    5, 2399,   11],
# [  20 , 292 ,2026 ,  11 , 128 ,   9 ,  12,   14 , 107, 2265]
# ]
# ) # huffu news
# dataset = 'HUFF-news'

# cm = np.array([[381,   0,   1 ,  0  , 4],
#  [  4 ,439 , 18 , 10,  39],
#  [  0  , 2 ,408  , 2 ,  5],
#  [  0 ,  1 ,  0 ,510 ,  0],
#  [  0  , 4 ,  0 ,  1 ,396]]
# ) # bbc
# dataset = 'BBC'



# cm = np.array([[188 ,  1 ,  3 ,  1],
#  [  1 ,146  ,52  , 1],
#  [  6 , 10, 182,   1],
#  [  0  , 0  , 0 ,198]]

# )  # Toutiao
# dataset = 'TOUTIAO'

# cm = np.array([[1427  , 82 , 364,  27],
#  [  11, 1799,   80 , 10],
#  [  69  , 55 ,1382 , 394],
#  [  35  , 33 , 479, 1353]]) # ag test
# dataset = 'AG-news-test'



# cm = np.array([[1060  , 29 ,  40],
#  [  27 ,1271  , 25],
#  [  63  , 23 , 918]] # mini news
# )
# dataset = 'HUFF-Mini-news'


# cm = np.array([[22740  , 968 , 5844  , 448],
#  [  317 ,27523 , 2047  , 113],
#  [ 1073  , 545 ,22475 , 5907],
#  [  714   ,256  ,5519 ,23511]] #ag train
# )
# dataset = 'AG-news-train'

# cm = np.array([[1241,    5,   61],
#  [  27,  987 ,   5],
#  [  27,    6, 1061]] # aminer-4v
# )
# dataset = 'AMiner-4v'


# labels_list = [i for i in range(len(cm))]
# title = 'cm-{}'.format(dataset)
# plot_confusion_matrix(cm, labels_list, title=title)
