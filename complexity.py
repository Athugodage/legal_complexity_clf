import numpy as np
from models import create_and_load_model, load_custom_model, ClusterCLF
from transformers import AutoTokenizer
from bert_text_prediction import single_pipeline, get_last_hidden_state_embedding
from main import get_arguments
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def complexity_by_border(embedding, clf_model):
    c = clf_model.model.predict(embedding.reshape(1, -1))[0]
    dist_to_center = np.linalg.norm(embedding - clf_model.model.cluster_centers_[c])
    return dist_to_center > clf_model.c_dist_quantiles_99[c]


def complexity_by_distance(embedding, clf_model, bins=10):
    dist_to_center = np.linalg.norm(clf_model.model.cluster_centers_ - embedding[None, :], axis=1)
    r = np.histogram(dist_to_center, bins=bins)[0][0]
    return r


def main():
    args = get_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    clf_model = create_and_load_model(args, DEVICE)
    cluster_model = load_custom_model(args, 'cluster')

    # get bert embeddings
    _, _, encoding, _ = single_pipeline(clf_model, tokenizer, args)
    bert_emb = get_last_hidden_state_embedding(encoding, clf_model).cpu().detach().numpy().reshape(-1)

    complexity = max(
        complexity_by_distance(bert_emb, cluster_model), complexity_by_border(bert_emb, cluster_model)
    )
    if complexity == 1:
        print('Простой')
    if complexity == 2:
        print('Сложный')
    if complexity > 2:
        print('Очень сложный')



if __name__ == '__main__':
    main()