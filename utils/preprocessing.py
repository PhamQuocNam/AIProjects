import torch
import pandas as pd
import re
from tokenizers import Tokenizer, trainers, pre_tokenizers, models
from transformers import AutoTokenizer

def text_normalize(text):
    text= re.sub('[^A-Za-z0-9]+', ' ', text)
    return text.lower()


def pad_and_truncate(sample, seq_len, pad_idx):
    if len(sample) >= seq_len:
        sample = sample[:seq_len]
    
    else:
        sample= sample + [pad_idx]*(seq_len-len(sample))
    
    return sample


def vectorize(data,tokenizer,seq_len,polarity_to_idx, model_name):
    term_inputs = []
    sentiment_inputs= []
    targets=[]
    polarities=[]
    for idx,sample in data.iterrows():
        # Tokenize sentence
        if 'bert' in model_name:
            tokenized = tokenizer(sample['Sentence'],return_offsets_mapping=True)
            term_input_ids = tokenized['input_ids']
            pad_idx = tokenizer.convert_tokens_to_ids('[PAD]')
        else:
            tokenized = tokenizer.encode(sample['Sentence'])# Assuming you use HuggingFace's `tokenizers` lib
            term_input_ids = tokenized.ids
            pad_idx = tokenizer.token_to_id('[PAD]')
        term_input = pad_and_truncate(term_input_ids, seq_len,pad_idx)
        
        # Sentiment input = sentence + [SEP] + aspect term
        if 'bert' in model_name:
            sentiment_input= term_input_ids + [tokenizer.convert_tokens_to_ids(word) for word in sample['Aspect Term']]
        else:
            aspect_ids = tokenizer.encode(sample['Aspect Term']).ids
            sentiment_input = term_input_ids + [tokenizer.token_to_id('[SEP]')] + aspect_ids
        sentiment_input = pad_and_truncate(sentiment_input, seq_len,pad_idx)
            
        if 'bert' in model_name:
            offsets = tokenized.offset_mapping
        else:
            offsets = tokenized.offsets
        start = sample['from']
        end = sample['to']
        size= min(len(offsets),len(term_input))
        idx_start = 0
        idx_end = size-1
        
        while offsets[idx_start][0] != start and idx_start<idx_end:
            if 'bert' in model_name:
                if tokenizer.convert_ids_to_tokens(term_input[idx_start]) in sample['Aspect Term']:
                    break
            elif tokenizer.id_to_token(term_input[idx_start]) in sample['Aspect Term']:
                break
            idx_start+=1

        while offsets[idx_end][1] != end and idx_start<idx_end:
            if 'bert' in model_name:
                if tokenizer.convert_ids_to_tokens(term_input[idx_start]) in sample['Aspect Term']:
                    break
            elif tokenizer.id_to_token(term_input[idx_start]) in sample['Aspect Term']:
                break
            idx_end-=1
            
        target = [1 if i>= idx_start and i<=idx_end else 0 for i in range(size)]
        target= pad_and_truncate(target, seq_len,-100)
        targets.append(target)
        polarity= polarity_to_idx[sample['polarity']]
        term_inputs.append(term_input)
        sentiment_inputs.append(sentiment_input)
        targets.append(target)
        polarities.append(polarity)
    
    return torch.tensor(term_inputs), torch.tensor(sentiment_inputs), torch.tensor(targets), torch.tensor(polarities)
    
    
   
def build_corpus(data_path):
    df = pd.read_csv(data_path)
    return df['Sentence'].apply(lambda x: text_normalize(x)).values
        


def build_tokenizer(corpus):
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]','[SEP]'])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(corpus, trainer)
    
    return tokenizer

def bert_tokenizer():
    return AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        

def get_seq_len(data):
    return max(len(text.split(' ')) for text in data)
        
        
     
        
        
        
        
        
        
    
    