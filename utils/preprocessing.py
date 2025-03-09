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


def term_vectorize(data, tokenizer, seq_len):
    input_ids=[]
    labels=[]
    
    for sentence, aspect_term in zip(data['sentence'],data['aspect_term']):
        input_id = [tokenizer.token_to_id(word) if tokenizer.token_to_id(word) else 0  for word in sentence.split(' ') ]
        input_id= pad_and_truncate(input_id,seq_len, tokenizer.token_to_id('<pad>'))
        
        key_labels = aspect_term.split(' ')
        label=[]
        for word in sentence.split(' '):
            if word in key_labels:
                if word == key_labels[0]:
                    label.append(1)
                else:
                    label.append(2)
            else:
                label.append(0)
                
        
        label = pad_and_truncate(label, seq_len, -100)
        input_ids.append(input_id)
        labels.append(label)
    return {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels),  
            }
    
    

def sentiment_vectorize(data, tokenizer, seq_len):
    input_ids=[]
    polarities=[]
    for sentence, aspect_term , polarity in zip(data['sentence'],data['aspect_term'] ,data['polarity']):
        sentence= sentence + ' <sep> ' + aspect_term
        
        input_id = [tokenizer.token_to_id(word) if tokenizer.token_to_id(word) else 0  for word in sentence.split(' ') ]
        input_id= pad_and_truncate(input_id,seq_len, tokenizer.token_to_id('<pad>'))
        
        if polarity=='neural':
            key_polarity = 0
        elif polarity=='positive':
            key_polarity =1
        else:
            key_polarity =2
        
        
        input_ids.append(input_id)
        polarities.append(key_polarity)
    
    return {
        'input_ids': torch.tensor(input_ids),
        'polarities': torch.tensor(polarities)    
            }
        
        
def build_corpus(data_path):
    df = pd.read_csv(data_path)
    return df.Sentence.apply(lambda x: text_normalize(x)).values
        


def build_tokenizer(corpus):
    
    model = models.WordLevel(unk_token='<unk>')
    tokenizer= Tokenizer(model)
    tokenizer.pre_tokenizer= pre_tokenizers.Whitespace()
    trainer=trainers.WordLevelTrainer(vocab_size=10000, special_tokens = ['<unk>','<pad>','<sos>','<eos>','<sep>'])
    tokenizer.train_from_iterator(corpus, trainer)
    
    return tokenizer

def bert_tokenizer():
    return AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        

def get_seq_len(data):
    return max(len(text.split(' ')) for text in data)
        
        
        
def term_bert_vectorize(data,tokenizer,seq_len):
    input_ids=[]
    labels=[]
    for sentence, aspect_term in zip(data['sentence'],data['aspect_term']):
        bert_tokens = []
        bert_tags = []
        key_aspect=tokenizer.tokenize(aspect_term)
        for word in sentence.split(' '):
            t = tokenizer.tokenize(word)
            bert_tokens += t
            if word in key_aspect:
                if word == key_aspect[0]:
                    bert_tags+= [1]*len(t)
                else:
                    bert_tags+= [2]*len(t)
            else:
                bert_tags+=[0]*len(t)
        bert_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
        bert_tokens = pad_and_truncate(bert_tokens,seq_len,tokenizer.convert_tokens_to_ids('<pad>'))
        bert_tags = pad_and_truncate(bert_tags,seq_len,-100)
        input_ids.append(bert_tokens)
        labels.append(bert_tags)
    return {
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels),  
            }
    
        
        
    
def sentiment_bert_vectorize(data,tokenizer,seq_len):
    input_ids=[]
    labels=[]
    for sentence, aspect_term , polarity in zip(data['sentence'], data['aspect_term'],data['polarity']):
        
        if polarity == 'neural':
            labels.append(0)
        elif polarity == 'positive':
            labels.append(1)
        else:
            labels.append(2)
        
        input_id = tokenizer(sentence,aspect_term)['input_ids']
        input_id = pad_and_truncate(input_id,seq_len,tokenizer.convert_tokens_to_ids('<padding>'))
        input_ids.append(input_id)
    
    
    return {
        'input_ids': torch.tensor(input_ids),
        'polarities': torch.tensor(labels),  
            }
    
 
        
        
        
        
        
        
    
    