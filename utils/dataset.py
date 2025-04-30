import pandas as pd
from .preprocessing import  text_normalize
from torch.utils.data import Dataset
import torch


class ABSADataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,idx):
        return self.inputs[idx], self.labels[idx]

    
def get_dataset(data_path):
    df= pd.read_csv(data_path)
    
    df['Sentence']= df['Sentence'].apply(lambda x: text_normalize(x))
    df['Aspect Term']= df['Aspect Term'].apply(lambda x: text_normalize(x))
    
    return df
    
    
    
    
    
    
    
    