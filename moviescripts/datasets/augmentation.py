from typing import List, Optional, Tuple, Union
import os, pickle, numpy as np, fasttext.util, nltk, random
from nltk.corpus import wordnet



class AddNoise:
    def __init__(self, p=0.2 ,dict_path = 'dict_of_words.pickle', cfg: Optional[str] = None):
        nltk.download('wordnet')

        self.dict_of_synonyms = self.load_dict_of_synonyms(dict_path)
        self.p = p
        self.cfg = cfg
    def load_dict_of_synonyms(self, data_path):
        if os.path.exists(data_path):
            with open(data_path, 'rb') as handle:
                dict_of_synonyms = pickle.load(handle)
        else:
            fasttext.util.download_model('en', if_exists='ignore')  # English
            ft = fasttext.load_model('cc.en.300.bin')
            words = np.unique([j for i in new_x for j in i.split()])
            dict_of_synonyms = {x: ft.get_nearest_neighbors(x) for x in words}
            with open(data_path, 'wb') as handle:
                pickle.dump(dict_of_synonyms, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return dict_of_synonyms    
    def __call__(self, text):
        words = text.split()
        num_noise_words = int(len(words) * self.p)
        for i in range(num_noise_words):
            idx = random.randint(0, len(words)-1)
            word = words[idx]
            synsets = wordnet.synsets(word)
            if synsets:
                synset = random.choice(synsets)
                synonyms = synset.lemmas()
                if random.random() < self.p/2:
                    synonym = random.choice(synonyms)
                    words[idx] = synonym.name()
                else:
                    if word in self.dict_of_synonyms:
                        nn = random.choice(self.dict_of_synonyms[word])
                        words[idx] = nn[1]
                    else:
                        synonym = random.choice(synonyms)
                        words[idx] = synonym.name()
        return ' '.join(words)
