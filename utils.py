import sys
import datetime

class DataLogger(object):
    def __init__(self, data, filename=None):
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if filename is None:
            filename = f"./exps/training/{data}_retrieval.log"
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass

import json
import os
def write_list_to_json(list, json_file_name):
    """
    将list写入到json文件
    :param list:
    :param json_file_name: 写入的json文件名字
    :return:
    """
    path = './training/'
    os.chdir(path)
    with open(json_file_name, 'w') as  f:
        json.dump(list, f)


import torch
import torch.nn as nn

def cosine_similarity_retrieval(query_vectors, document_vectors, k=5):
    """
    对于每个查询向量，计算其与一组文档向量之间的余弦相似度，并返回排序后的相似度分数和文档索引。

    参数:
    query_vectors (torch.Tensor): 查询向量的集合。
    document_vectors (torch.Tensor): 文档向量的集合。
    k (int): 返回的最大结果数。

    返回:
    dict: 对于每个查询的排序后的文档索引和相似度分数的字典。
    """
    # 初始化余弦相似度对象
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # 存储每个查询的结果
    all_results = {}

    for i, query_vector in enumerate(query_vectors):
        # 计算相似度
        similarity_scores = cos(query_vector.unsqueeze(0), document_vectors)

        # 对相似度进行排序
        sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)

        # 仅保留前k个结果
        sorted_scores, sorted_indices = sorted_scores[:k], sorted_indices[:k]

        # 将结果组合成元组列表（索引，分数）
        ranked_results = [(index.item(), score.item()) for index, score in zip(sorted_indices, sorted_scores)]

        # 将结果添加到字典中
        all_results[i] = ranked_results

    return all_results


def matching_acc(a, b, k):

    matching_count = 0
    results = cosine_similarity_retrieval(a, b, k)

    # 打印结果
    for query_index, ranked_documents in results.items():
        # print(f"Query {query_index}:")
        # for doc_index, score in ranked_documents:
        #     print(f"  Document {doc_index}: Similarity Score = {score}")
        doc_index  = [item[0] for item in ranked_documents]
        if query_index in doc_index:
            matching_count += 1
    
    return matching_count / len(results)

