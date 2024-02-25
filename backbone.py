from sentence_transformers import SentenceTransformer
from torch import nn
from gensim import corpora, models
from preprocessing import preprocessing
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentenceTransformerBackbone(nn.Module):
    def __init__(self, model_name_or_path='distiluse-base-multilingual-cased-v1'):
        super(SentenceTransformerBackbone, self).__init__()
        self.model = SentenceTransformer(model_name_or_path)
    
    def forward(self, x):
        return self.model.encode(x)
    
    def get_embedding_dim(self):
        return self.model.get_sentence_embedding_dimension()
    
    def get_config_dict(self):
        return self.model.get_config_dict()
    
class TFIDFBackbone(nn.Module):
    def __init__(self):
        super(TFIDFBackbone, self).__init__()
        
            
    def forward(self, texts):
        # 创建语料库
        self.dictionary = corpora.Dictionary([text.split() for text in texts])
        # 将文本转换为文档-词矩阵
        self.corpus = [self.dictionary.doc2bow(text.split()) for text in texts]

        self.model = models.TfidfModel(self.corpus)
        return self.model[self.corpus]
    
    def get_embedding_dim(self):
        return len(self.dictionary)
    
class TFIDFBackbone_SK(nn.Module):
    def __init__(self):
        super(TFIDFBackbone_SK, self).__init__()

    def forward(self, texts, type):
        return preprocessing(text_list=texts, type=type)


def get_backbone(type=0):
    if type == 0:
        return TFIDFBackbone_SK()
    elif type == 1:
        return SentenceTransformerBackbone().to(device)
    else:
        raise ValueError('type must be 0 or 1')