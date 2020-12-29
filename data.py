from torch.utils.data import Dataset
import torch
from sklearn import preprocessing
import os
import pandas as pd

class FemDataset(Dataset):
    def __init__(self, file_name):
        file_path = os.path.join('data', file_name)
        self.df = pd.read_csv(file_path, header=0)
        self.preprocess()
    def preprocess(self):
        self.df = self.df.drop('Sample', axis=1)
        columnsX = [
            'ecc',
            'N',
            'gammaG',
            'Esoil',
            'Econc',
            'Dbot',
            'H1',
            'H2',
            'H3'
        ]
        columnsY = [
            'Mr_t',
            'Mt_t',
            'Mr_c',
            'Mt_c'
        ]
        self.X = self.df[columnsX]
        self.y = self.df[columnsY]
        self.X = preprocessing.scale(self.X)
        self.y = preprocessing.scale(self.y)
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)
    def __len__(self):
        return(len(self.X))
    def __getitem__(self, i):
        return self.X[i], self.y[i]
