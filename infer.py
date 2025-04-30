import torch
import torch.nn as nn
from utils.preprocessing import build_tokenizer, build_corpus, bert_tokenizer,\
     text_normalize, pad_and_truncate
from Models.base_model import Term_ABSAModel,Sentiment_ABSAModel
from Models.bert import Term_BertModel,Sentiment_BertModel
from transformers import AutoModelForTokenClassification
import argparse
from utils.dataset import get_dataset
class inference:
    def __init__(self, opt):
        self.opt= opt
        if 'bert' in opt.model_name:
            bert = AutoModelForTokenClassification.from_pretrained(opt.pretrained_bert_name)
            self.tokenizer =  bert_tokenizer()
            
            self.term_model = opt.model_class[0](bert,opt).to(opt.device)
            self.sentiment_model = opt.model_class[1](bert,opt).to(opt.device)
            
        else: 
            self.tokenizer = build_tokenizer(build_corpus(opt.dataset_file['train']))
            self.opt.vocab_size= self.tokenizer.get_vocab_size()
        
            self.term_model = opt.model_class[0](opt).to(opt.device)
            self.sentiment_model = opt.model_class[1](opt).to(opt.device)
        self.term_model.load_state_dict(torch.load('checkpoints/term_model.pth'))
        self.sentiment_model.load_state_dict(torch.load('checkpoints/sentiment_model.pth'))
        
        self.term_model.eval()
        self.sentiment_model.eval()
        torch.autograd.set_grad_enabled(False)
        
    def evaluate(self,text):
        self.term_model.eval()
        self.sentiment_model.eval()
        text= text_normalize(text)
        print('Normalize :', text)
        if 'bert' in self.opt.model_name:
            tokenized = self.tokenizer(text,return_offsets_mapping=True)
            term_input_ids = tokenized['input_ids']
            pad_idx = self.tokenizer.convert_tokens_to_ids('[PAD]')
        else:
            tokenized = self.tokenizer.encode(text)# Assuming you use HuggingFace's `tokenizers` lib
            term_input_ids = tokenized.ids
            pad_idx = self.tokenizer.token_to_id('[PAD]')
        term_input = pad_and_truncate(term_input_ids, self.opt.seq_len,pad_idx)
        term_input = torch.tensor(term_input,device= self.opt.device).unsqueeze(0)
        output = self.term_model(term_input).argmax(1)[0]
        
        aspect_term =[]
        for i in range(len(tokenized)):
            if output[i] != 0:
                aspect_term.append(term_input[-1][i])
        
        if 'bert' in self.opt.model_name:
            sentiment_input = term_input_ids + aspect_term
        else:
            sentiment_input = term_input_ids + [self.tokenizer.token_to_id('[SEP]')] + aspect_term
            
        sentiment_input = pad_and_truncate(sentiment_input, self.opt.seq_len,pad_idx)
        sentiment_input = torch.tensor(sentiment_input,device= self.opt.device).unsqueeze(0)
        
        output = self.sentiment_model(sentiment_input).argmax(1)[0]
        polarity = self.opt.idx_to_polarity[int(output)]
        
        if 'bert' in self.opt.model_name:
            aspect_term = ' '.join(self.tokenizer.convert_ids_to_tokens(aspect_term))
        else:
            aspect_term = ' '.join([self.tokenizer.id_to_token(id) for id in aspect_term])
        
        print(f'Aspect term: {aspect_term}, Polarity: {polarity}')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_model', type=str)
    parser.add_argument('--pretrained_bert_name', default='distilbert/distilbert-base-uncased', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    
    model_classes = {
        'base_model': [Term_ABSAModel,Sentiment_ABSAModel],
        'bert_model': [Term_BertModel,Sentiment_BertModel]
    }
    
    dataset_files = {
        'restaurant': {
            'train': 'Dataset/Restaurants_Train_v2.csv',
            'val': 'ABSA/Dataset/restaurants-trial.csv'
        },
        'laptop': {
            'train': 'Dataset/Laptop_Train_v2.csv',
            'val': 'Dataset/laptops-trial.csv'
        }
    }
    
    opt = parser.parse_args()
    
    
    data = get_dataset(dataset_files[opt.dataset]['train'])
    polarity_unique=data['polarity'].unique()
    opt.polarity_to_idx = {old : new for new,old in enumerate(polarity_unique)}
    opt.idx_to_polarity = {new : old for old, new in opt.polarity_to_idx.items()}
    
    opt.model_name = 'bert_model'
    opt.model_class = model_classes[opt.model_name]
    
    # set your trained models here
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.term_seq_len = 100
    opt.sentiment_seq_len=120
    opt.bert_dim = 768

    opt.pretrained_bert_name = 'distilbert/distilbert-base-uncased'
    opt.term_num_classes=3
    opt.sentiment_num_classes = 4
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.seq_len=80
    

    inf = inference(opt)
    inf.evaluate('this laptop run so fast')
                                         
        
        
        
        
        
        