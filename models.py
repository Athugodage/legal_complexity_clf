import numpy as np
from transformers import AutoModel
import torch
from torch import nn
import pickle
from scipy.special import softmax

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased'


class KADClassifier(nn.Module):
    def __init__(self, n_classes=35, model_name=PRE_TRAINED_MODEL_NAME):
        super(KADClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(outputs["pooler_output"])
        return self.out(output)


def create_and_load_model(args, device='cuda'):
    clf_model = KADClassifier(model_name=args.bert_path).to(device)
    return clf_model


class ClusterCLF:
    def __init__(self, model, cluster_to_category, cluster_coef, c_dist, n_classes=6):
        self.model = model
        self.cluster_to_category = cluster_to_category
        self.n_classes = n_classes
        self.cluster_coef = cluster_coef
        self.c_dist_quantiles_99 = c_dist

    def get_category_proba(self, embeddings):
        cluster_dist = np.array([np.linalg.norm(emb - self.model.cluster_centers_, axis=1) for emb in embeddings])
        self.category_dist = np.full((cluster_dist.shape[0], self.n_classes), float('inf'))
        for c, category in self.cluster_to_category.items():
            self.category_dist[:, category] = np.minimum(self.category_dist[:, category], cluster_dist[:, c])
        return softmax(-self.category_dist, axis=1), self.category_dist

    def complexity_by_distance(self, embeddings, bins=10):
        dist_to_center = np.linalg.norm(np.array(self.model.cluster_centers_) - np.array(embeddings[None, :]), axis=1)
        r = np.histogram(dist_to_center, bins=bins)[0][0]
        return r, dist_to_center

    def complexity_by_border(self, embedding):
        c = self.model.predict(embedding.reshape(1, -1))[0]
        dist_to_center = np.linalg.norm(embedding - self.model.cluster_centers_[c])
        return dist_to_center > self.c_dist_quantiles_99[c]

    def predict(self, X):
        clust_distances = []
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        self.proba, category_dist = self.get_category_proba(X)
        clust_distances.append(category_dist)
        pred = self.proba.argmax(axis=1)
        return pred, self.proba, clust_distances


def load_custom_model(args, mode='cluster'):
    if mode == 'cluster':
        path = args.cluster_path
    elif mode == 'graph':
        path = args.graph_path
    with open(path, 'rb') as f:
        custom_model = pickle.load(f)
    print(f"{mode} model loaded", flush=True)
    return custom_model


class GraphModel:
    def __init__(self, graph_data, graph_coef=1e-4):
        graph_data.sort_values('bert_id', inplace=True)
        n_features_per_category = graph_data.groupby('CategoryID')['word'].count()
        assert n_features_per_category.unique().shape[0] == 1
        self.n_features_per_category = n_features_per_category.unique()[0]
        self.graph_keywords = graph_data.word
        self.bert_ids = graph_data.bert_id.unique()
        self.graph_coef = graph_coef

    def proba(self, tokens):
        self.w_embs = np.array([1 if word in tokens else 0 for word in self.graph_keywords])
        self.w_embs = self.w_embs.reshape(-1, self.n_features_per_category)
        proba = softmax(self.w_embs.mean(axis=1))
        return proba

    def predict(self, tokens):
        proba = self.proba(tokens)
        return self.bert_ids[proba.argmax()]
