
from fire import Fire
from loguru import logger
import multiprocessing

import os
import pandas as pd
import numpy as np
class StarWarsPreprocessing:
    def __init__( self,
        data_dir: str = "./data/raw/StarWarsEpisodes",out_dir = "./data/raw/starwars"
        ) -> None:
        self.data_dir = data_dir
        self.out_dir = out_dir

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
            if b < 10:
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

if __name__ == "__main__":
    Fire(StarWarsPreprocessing)