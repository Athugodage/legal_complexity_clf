import pandas as pd
import numpy as np
from collections import Counter

import argparse
import json
import io
import base64
import configparser
import gdown

from transformers import AutoTokenizer
from sklearn.cluster import KMeans

import torch
from models import create_and_load_model, load_custom_model, ClusterCLF, GraphModel
from bert_text_prediction import single_pipeline, get_last_hidden_state_embedding
from text_preprocessing import prepare_russian_text, replace_org, Speller
from classification import ComplexityClassifier

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance

import logging

config = configparser.ConfigParser()
config.read('config.ini')
DEVICE = 'cpu'

gdown.download_folder(config['GENERAL']['disk_url'])

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


ft_dict = {
    "ЭС": "экономические споры",
    "КС": "корпоративные споры",
    "АП": "административное правоотношение",
    "ТБ": "несостоятельность (банкротство)",
    "ТС": "третейский суд",
    "ИС": "иностранный суд"
}

colors = {'ЭС': 'maroon', '-': 'olive',
          'ТБ': 'aquamarine', 'АП': 'teal', 'КС': 'salmon',
          'ИС': 'magenta',
          'ТС': 'gold'}

DEVICE = 'cpu'


def __compose_dic__(names, probas):
    d = {}

    for n in range(len(names)):
        name = names[n]

        proba = probas[n]
        d[name] = proba

    return d


def __forreport__(dic, complexity):
    nearest_clusters = []

    dic = Counter(dic)
    if complexity == 'Простой':
        top = 1
    elif complexity == 'Сложный':
        top = 2
    else:
        top = 3
    for name, value in dic.most_common(top):
        nearest_clusters.append(name)

    return nearest_clusters

# clusterplot ->
# barplot -> result_d, xlabels


class BlackBox():
    def __init__(self, args):
        self.args = args
        logging.info('Now we start loading models and datasets')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_path)
        self.clf_model = create_and_load_model(self.args, DEVICE)
        self.cluster_model = load_custom_model(self.args, 'cluster')
        self.graph_model = load_custom_model(self.args, 'graph')
        self.categs_info = pd.read_csv(self.args.saved_data_path + f'/{args.kad_path}')  # map bert id to category id
        self.leave_org = np.load(f'{self.args.saved_data_path}/leave_org.npy')  # for text preprocessing
        self.feature_phrase = pd.read_csv(f'{self.args.saved_data_path}/feature_phrase.csv')  # for features
        self.df_kat = pd.read_csv(config['OLD_PIPE']['df_kat'])


        self.new_clf = ComplexityClassifier()
        self.speller = Speller(model=config['GENERAL']['speller'])

        logging.info('Loaded all models! Prepared to process text.')

        self.label2id = {'легкий': 0, 'сложный': 1, 'очень сложный': 2}
        self.id2label = {0: 'легкий', 1: 'сложный', 2: 'очень сложный'}



    def summ_probas(self, probas):
        spori = self.df_kat['Descr'].to_list()
        d = {}
        voc = {}
        result_d = {}

        for i in range(len(probas)):
            d[spori[i]] = probas[i]

        for i in range(len(self.df_kat)):
            spor = self.df_kat.iloc[i]['Descr']
            kat = self.df_kat.iloc[i]['features']
            if kat not in voc.keys():
                voc[kat] = []
            voc[kat].append(spor)

        for k in voc.keys():
            score = 0
            for spor in voc[k]:
                score += float(d[spor])
            result_d[k] = score

        return result_d

    def complexity_by_border(self, embedding):
        c = self.cluster_model.model.predict(embedding.reshape(1, -1))[0]
        dist_to_center = np.linalg.norm(embedding - self.cluster_model.model.cluster_centers_[c])
        return dist_to_center > self.cluster_model.c_dist_quantiles_99[c]


    def complexity_by_distance(self, embedding, bins=10):
        dist_to_center = np.linalg.norm(self.cluster_model.model.cluster_centers_ - embedding[None, :], axis=1)
        r = np.histogram(dist_to_center, bins=bins)[0][0]
        return r


    def complexity_main(self, text):
        # get bert embeddings
        _, _, encoding, _ = single_pipeline(self.clf_model, self.tokenizer, text)
        bert_emb = get_last_hidden_state_embedding(encoding, self.clf_model).cpu().detach().numpy().reshape(-1)

        complexity = max(
            self.complexity_by_distance(bert_emb),
            self.complexity_by_border(bert_emb)
        )
        if complexity < 2:
            return 'Простой'
        elif complexity == 2:
            return 'Сложный'
        else:
            return 'Очень сложный'


    def get_barplot_info(self, complexity):
        ## DATA PREPARATION

        with open(self.args.save_report, 'r') as f:
            f = json.load(f)
        probas = f['report']['probability']
        names = self.df_kat['Descr'].to_list()
        dic = __compose_dic__(names, probas)
        nearest_clusters = __forreport__(dic, complexity)

        result_d = self.summ_probas(probas)  # dist_to_center[:35]

        ## MAKE PLOT
        xlabels_new = list(map(lambda x: x.replace('-', 'Другое'), result_d.keys()))


        return result_d, xlabels_new, nearest_clusters
        ## result a tuple of data for plot and its labels. Should be added to json as two indepedent values.



    def get_features(self, pred_cat_info, tokens, feature_phrase):
        bigrams = [f'{w1} {w2}' for w1, w2 in (zip(tokens[:-1], tokens[1:]))]
        features_info = {k: False for k in ft_dict.keys()}
        if pred_cat_info.features in features_info:
            features_info[pred_cat_info.features] = True
        feature_phrase['feature_occurrence'] = np.array([phrase in bigrams for phrase in feature_phrase.phrase])
        occurrence_per_feature = feature_phrase.groupby('feature')['feature_occurrence'].sum()
        for ft in occurrence_per_feature[occurrence_per_feature != 0].index:
            features_info[ft] = True
        return features_info

    def category_item(self, bert_id_pred, categs_info, proba, tokens):
        pred_cat_info = self.categs_info.query('bert_id == @bert_id_pred').iloc[0]
        features_info = self.get_features(pred_cat_info, tokens, self.feature_phrase)
        return {
            'KAD_ID': int(pred_cat_info.KAD_ID),
            'Description': pred_cat_info.Descr,
            'probability': proba
        } | features_info


    def complexity2(self, text):
        try:
            label = self.new_clf.predict(text)
            logging.info('Made prediction by our new LSTM classefier')
            logging.info(label)

        except:
            label = None
            logging.warning('Some problems with predicting. Model loaded but does not respond.')
            logging.warning('We put default label - "None". But in fact we failed to make prediction.')

        return label


    def complexity1(self, text):
        # combined model prediction
        _, bert_proba, encoding, cleaned_text = single_pipeline(self.clf_model, self.tokenizer, text)
        bert_emb = get_last_hidden_state_embedding(encoding, self.clf_model)
        _, cluster_proba, clust_distances = self.cluster_model.predict(bert_emb)
        text_money_org_replaced = replace_org(cleaned_text, self.leave_org)
        preprocessed_tokens = prepare_russian_text(text_money_org_replaced)
        graph_proba = self.graph_model.proba(preprocessed_tokens).reshape(1, -1)
        by_distance, dist_to_center = self.cluster_model.complexity_by_distance(bert_emb)
        # by_border = cluster_model.complexity_by_border(bert_emb)

        # complexity = by_distance

        combo_proba = (bert_proba + self.cluster_model.cluster_coef * cluster_proba + self.graph_model.graph_coef * graph_proba
                      ) / (1 + self.cluster_model.cluster_coef + self.graph_model.graph_coef)
        combo_proba = combo_proba.cpu().detach().numpy()
        combo_pred = combo_proba.argmax(axis=1).item()
        self.res = self.category_item(
            combo_pred, self.categs_info, combo_proba.reshape(-1).tolist(),
            preprocessed_tokens)

        complexity = self.complexity_main(text)

        logging.info('Found complexity by clusters')
        return complexity, by_distance, dist_to_center, clust_distances

    def cluster_plot(self):
        embeddings = np.load(config['OLD_PIPE']['embeddings'])
        kmeans = KMeans(n_clusters=6)
        visualizer = InterclusterDistance(kmeans, min_size=0.1, legend=False,
                                          classes=["ЭС", "КС", "АП", "ТБ", "ТС", "ИС"],
                                          title='Intercluster distance',
                                          colors="teal")  # KElbowVisualizer(km_cl, k=(1,7))
        visualizer.fit(embeddings)
        # plt.scatter(1, 3, s=300, c="blue", alpha=0.4, linewidth=3)
        visualizer.show(outpath=config['OLD_PIPE']['cluster_plot'])

        img = mpimg.imread(config['OLD_PIPE']['cluster_plot'])
        plt.imshow(img)

        s = io.BytesIO()
        plt.savefig(s, format='png', bbox_inches="tight")
        plt.close()
        s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
        return '<img align="left" src="data:image/png;base64,%s">' % s


    def make_review(self, complexity, by_distance, dist_to_center, clust_distances):
        plot_description = '''0 - экономические споры\n 1-корпоративные споры\n
                                  2 - административное правоотношение\n 3 - несостоятельность (банкротство)\n
                                  4 - третейский суд\n 5 - иностранный суд'''

        categs_in = []
        for i in self.res.keys():
            if i in ['ЭС', 'КС', 'АП', 'ТБ', 'ТС', 'ИС']:
                if self.res[i] == True:
                    categs_in.append(i)

        categs_in = [ft_dict[categ] for categ in categs_in]

        result_dic = {'report': self.res,  ## all initial information from the previous worker
                      'complexity_class': complexity,  ## predict_complexity()
                      'distances': clust_distances,
                      'cluster_centres': self.cluster_model.model.cluster_centers_,
                      'cluster_plot_description': plot_description
                      }

        result_dic1 = {'cluster_distances': clust_distances,
                       'dist_to_center': dist_to_center}

        with open(self.args.save_report, 'w') as w:
            json.dump(result_dic, w, cls=NumpyEncoder)

        with open('cluster_complexity.json', 'w') as w:
            json.dump(result_dic1, w, cls=NpEncoder)

        barplot_data, barplot_xlabels, nearest_clusters = self.get_barplot_info(complexity)  # data for barplot. ADD it to json
        clsuter_plot = self.cluster_plot()  # draw cluster plot

        report0 = 'Данный документ можно классифицировать как '  # next: complexity.lower()
        report1 = ', т.к. он входит в следующие кластера: '  # next: ', '.join(nearest_clusters)
        report2 = 'Данный документ принадлежит следующим категориям:\n'  # next: ', '.join(categs_in)
        report3 = '''.
                На графике ниже вы видете какие кластера с какой вероятностью можно отнести к данному тексту. 

                Нотабене. Ниже представлена расшифровка аббревиатур:
                ЭС - экономические споры,
                КС - корпоративные споры,
                АП - административное правоотношение,
                ТБ - несостоятельность (банкротство),
                ТС - третейский суд,
                ИС - иностранный суд'''

        text_report = {0: report0, 1: complexity.lower(), 2: report1, 3: ', '.join(nearest_clusters),
                       4: report2, 5: ', '.join(categs_in), 6: report3}

        res = {'text_report': text_report,
               'barplot_data': barplot_data,
               'barplot_labels': barplot_xlabels,
               'cluster_plot': clsuter_plot,
               'result_dic': result_dic,
               'result_dic1': result_dic1}

        with open('res.json', 'w') as w:
            json.dump(res, w, cls=NumpyEncoder)



    def implement(self):
        with open(self.args.input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            logging.info("Opened file")

        text = self.speller.correct(text)

        label1, by_distance, dist_to_center, clust_distances = self.complexity1(text)

        label1 = self.label2id[label1.lower()]
        label2 = self.label2id[str(self.complexity2(text)).lower()]
        final_label = self.id2label[round(np.mean([label1, label2]))]

        self.make_review(final_label, by_distance, dist_to_center, clust_distances)

        return final_label






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=config['OLD_PIPE']['input_path'])
    parser.add_argument('--bert_path', type=str, default=config['OLD_PIPE']['bert_path'])
    parser.add_argument('--cluster_path', type=str, default=config['OLD_PIPE']['cluster_model'])
    parser.add_argument('--graph_path', type=str, default=config['OLD_PIPE']['graph_model'])
    parser.add_argument('--saved_data_path', type=str, default="saved_data")
    parser.add_argument('--cp_path', type=str, default=config['OLD_PIPE']['model_state'])
    parser.add_argument('--kad_path', type=str)
    parser.add_argument('--save_report', type=str, default=config['OLD_PIPE']['full_report'])
    parser.add_argument('--save_prediction_path', type=str, default=config['OLD_PIPE']['prediction_file'])

    args = parser.parse_args()
    KAD_CAT_INFO_PATH = args.kad_path

    pipe = BlackBox(args)
    result = pipe.implement()
    print(result)


