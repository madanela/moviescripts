
from fire import Fire
from loguru import logger
import multiprocessing
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
import random

from sentence_transformers import SentenceTransformer
from moviescripts.datasets.augmentation import AddNoise


class StarWarsPreprocessing:
    def __init__( self,
        data_dir: str = "data/raw/starwars",out_dir = "data/processed/starwars"
        ) -> None:
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        logger = logging.getLogger(__name__)

        if not self.data_dir.exists():
            logger.error("data folder doesn't exist")
            raise FileNotFoundError
        
        if self.out_dir.exists() is False:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.add_noise = None
    @logger.catch
    def preprocess(self):
        folder_ep4 = os.path.join(self.data_dir,"SW_EpisodeIV.txt")
        folder_ep5 = os.path.join(self.data_dir,"SW_EpisodeV.txt")
        folder_ep6 = os.path.join(self.data_dir,"SW_EpisodeVI.txt")
        df_ep4 = pd.read_csv(folder_ep4, sep =' ', header=0, escapechar='\\')
        df_ep5 = pd.read_csv(folder_ep5, sep =' ', header=0, escapechar='\\')
        df_ep6 = pd.read_csv(folder_ep6, sep =' ', header=0, escapechar='\\')
        Y = pd.concat([df_ep4['character'],df_ep5['character'],df_ep6['character']]).tolist()
        X = pd.concat([df_ep4['dialogue'],df_ep5['dialogue'],df_ep6['dialogue']]).tolist()
        labels = np.unique(Y)
        label_count = [sum(i == np.array(Y)) for i in labels]
        for i,(a,b) in enumerate(zip(labels,label_count)):
            if b < 40:
                labels[i] = "Other"
        labels = np.unique(labels)
        char2ind = {i:j for i,j in zip(labels,range(len(labels)))}
        ind2char = {j:i for i,j in zip(labels,range(len(labels)))}
        new_x = X.copy()
        new_y = []
        for idx in range(len(new_x)):
            
            if Y[idx] in labels:
                label_point = char2ind[Y[idx]]
            else:
                label_point = char2ind["Other"]
            new_y.append(label_point)
        dict_path = self.out_dir /'dict_of_words.pickle'
        self.add_noise = AddNoise(new_x = new_x,dict_path = dict_path,p = .7)


        # create the DataFrame
        df1 = pd.DataFrame({"X": new_x, "y": new_y})

        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(df1["X"], df1["y"], test_size=0.2, random_state=42)

        # create DataFrames for train and test sets
        train_df = pd.DataFrame({"X": X_train, "y": y_train})
        test_df = pd.DataFrame({"X": X_test, "y": y_test})

        # save train and test sets as separate CSV files
        train_df.to_csv(self.out_dir /"train.csv", index=False)
        test_df.to_csv(self.out_dir /"val.csv", index=False)


        path = Path(self.out_dir / "encoded_bertdata_train.csv")

        def str_to_float_list(s):
            float_list = [float(x) for x in re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d+|[-+]?\d+', s)]
            return float_list

        if not path.exists():
            train_encoded_X = []
            train_encoded_X_y = []
            val_encoded_X = []
            val_encoded_X_y = []

            temp = list(zip(new_x, new_y))
            random.shuffle(temp)
            new_x, new_y = zip(*temp)
            # res1 and res2 come out as tuples, and so must be converted to lists.
            new_x, new_y = list(new_x), list(new_y)
            
            model = SentenceTransformer('bert-base-nli-mean-tokens')

            #number of random tests
            for _ in range(10):
                for x,y in zip(new_x[:2000],new_y[:2000]):
                    x = model.encode(self.add_noise(x))
                    train_encoded_X.append(x)
                    train_encoded_X_y.append(y)
                #number of random tests
            for x,y in zip(new_x[2000:],new_y[2000:]):
                x = model.encode(x)
                val_encoded_X.append(x)
                val_encoded_X_y.append(y)
            pd.DataFrame({"X":train_encoded_X,"y":train_encoded_X_y}).to_csv(self.out_dir / "encoded_bertdata_train.csv")
            pd.DataFrame({"X":val_encoded_X,"y":val_encoded_X_y}).to_csv(self.out_dir / "encoded_bertdata_val.csv")
        else:
            df_train = pd.read_csv(self.out_dir / "encoded_bertdata_train.csv")     
            df_val = pd.read_csv(self.out_dir / "encoded_bertdata_val.csv")        
        
            train_encoded_X = df_train["X"].apply(str_to_float_list)
            train_encoded_X_y = df_train["y"]
            val_encoded_X = df_val["X"].apply(str_to_float_list)
            val_encoded_X_y = df_val["y"]


if __name__ == "__main__":
    Fire(StarWarsPreprocessing)