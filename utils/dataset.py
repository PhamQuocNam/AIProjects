import pandas as pd
from .preprocessing import  text_normalize
from torch.utils.data import Dataset

class Term_ABSADataset(Dataset):
    def __init__(self,data , term_vectorize, tokenizer, seq_len):
        temp= term_vectorize(data, tokenizer, seq_len)
        self.input_ids= temp['input_ids']
        self.labels = temp['labels']
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
    


class Sentiment_ABSADataset(Dataset):
    def __init__(self,data ,sentiment_vectorize , tokenizer, seq_len):
        temp= sentiment_vectorize(data, tokenizer, seq_len)
        self.input_ids= temp['input_ids']
        self.polarities= temp['polarities']
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.polarities[idx]
    
    
def get_dataset(data_path):
    df= pd.read_csv(data_path)
    
    dataset = {}
    dataset['sentence']= df.Sentence.apply(lambda x: text_normalize(x))
    dataset['aspect_term']= df['Aspect Term'].apply(lambda x: text_normalize(x))
    dataset['polarity']= df.polarity
    
    return dataset
    
    
    
    
    
    
    
    