from tensorflow.keras.layers import TextVectorization
import pickle
import keras
import numpy as np
import pandas as pd
import configparser
import gdown
import zipfile

import logging


config = configparser.ConfigParser()
config.read('config.ini')



class ComplexityClassifier():
    def __init__(self,
                 lstm_path=config['CLASSIFIER']['model'], #'/content/drive/MyDrive/complexity_lstm28052024/legal_complexity_LSTM.keras'
                 vectorizer_path=config['CLASSIFIER']['vectorizer'],
                 solo_pipe=False,
                 disk_path=config['GENERAL']['disk_url']):

        self.label2id = {'легкий': 0, 'сложный': 1, 'очень сложный': 2}
        self.id2label = {0: 'легкий', 1: 'сложный', 2: 'очень сложный'}

        if solo_pipe==True:
            gdown.download_folder(disk_path)




        self.model = keras.models.load_model(lstm_path)
        self.from_disk = pickle.load(open(vectorizer_path, "rb"))
        self.vectorizer = TextVectorization.from_config(self.from_disk['config'])

        self.df = pd.read_parquet(config['CLASSIFIER']['dataset'])
        self.X = self.df['text']
        self.vectorizer.adapt(self.X.values)
        logging.info('Loaded necessary files: model, vectorizer, dataset')





    def predict(self, text):
        text = str(text)
        label = np.argmax(self.model.predict(self.vectorizer([text, '']))[0])
        label = self.id2label[label]

        return label


