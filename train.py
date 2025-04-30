import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from torch.utils.data import Dataset, DataLoader , random_split
from tqdm import tqdm
from sklearn.metrics import f1_score
import random
import logging
import argparse
import os
from utils.preprocessing import build_tokenizer, build_corpus, get_seq_len, bert_tokenizer, vectorize
from utils.dataset import ABSADataset, get_dataset
from utils.metrics import compute_metric
from Models.base_model import Term_ABSAModel,Sentiment_ABSAModel
from Models.bert import Term_BertModel,Sentiment_BertModel
from transformers import AutoModelForTokenClassification



logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt= opt
        train_ds = get_dataset(opt.dataset_file['train'])
        val_ds = get_dataset(opt.dataset_file['val'])
        polarity_unique=train_ds['polarity'].unique()
        polarity_to_idx = {old : new for new,old in enumerate(polarity_unique)}
        idx_to_polarity = {new : old for old, new in polarity_to_idx.items()}
        opt.sentiment_num_classes=len(polarity_to_idx)
        if 'bert' in opt.model_name:
            bert = AutoModelForTokenClassification.from_pretrained(opt.pretrained_bert_name)
            self.tokenizer =  bert_tokenizer()            
            
        else: 
            self.tokenizer = build_tokenizer(build_corpus(opt.dataset_file['train']))
            self.opt.vocab_size= self.tokenizer.get_vocab_size()
            
        self.opt.term_seq_len = opt.term_seq_len
        self.opt.sentiment_seq_len = opt.sentiment_seq_len
            
        term_inputs, sentiment_inputs, targets, polarities= vectorize(train_ds,self.tokenizer, self.opt.term_seq_len,polarity_to_idx, opt.model_name)
        
        self.term_train_ds = ABSADataset(term_inputs, targets)
        self.sentiment_train_ds = ABSADataset(sentiment_inputs,polarities)
        
        term_inputs, sentiment_inputs, targets, polarities= vectorize(val_ds,self.tokenizer, self.opt.sentiment_seq_len,polarity_to_idx,opt.model_name)
        
        self.term_val_ds = ABSADataset(term_inputs, targets)
        self.sentiment_val_ds = ABSADataset(sentiment_inputs,polarities)
        if bert:
            self.term_model = opt.model_class[0](bert,opt).to(opt.device)
            self.sentiment_model = opt.model_class[1](bert,opt).to(opt.device)
        else:
            self.term_model = opt.model_class[0](opt).to(opt.device)
            self.sentiment_model = opt.model_class[1](opt).to(opt.device)
        print(torch.cuda.is_available())
        if opt.device == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        
    
    def run(self):
        criterion = nn.CrossEntropyLoss()
        term_optimizer = self.opt.optimizer(self.term_model.parameters(), lr=self.opt.lr)
        sentiment_optimizer = self.opt.optimizer(self.sentiment_model.parameters(),lr=self.opt.lr)
        term_train_data_loader = DataLoader(dataset=self.term_train_ds, batch_size=self.opt.batch_size, shuffle=True)
        sentiment_train_data_loader = DataLoader(dataset = self.sentiment_train_ds, batch_size = self.opt.batch_size, shuffle=True)
        
        term_val_data_loader = DataLoader(dataset=self.term_val_ds, batch_size=self.opt.batch_size, shuffle=False)
        sentiment_val_data_loader = DataLoader(dataset = self.sentiment_val_ds, batch_size = self.opt.batch_size, shuffle=False)
        
        self._train(criterion, self.term_model , term_optimizer, term_train_data_loader, term_val_data_loader,'checkpoints/term_model.pth')
        self._train(criterion, self.sentiment_model, sentiment_optimizer, sentiment_train_data_loader, sentiment_val_data_loader,
                                                'checkpoints/sentiment_model.pth')
        
        
        
    
    
    def _train(self, criterion, model, optimizer, train_loader, val_loader, file_path):
        total_losses=[]
        score_record=0
        
        best_params= model.state_dict()
        for epoch in tqdm(range(self.opt.epochs),desc='Epoch'):
            epoch_loss=[]
            model.train()
            for idx, (input_ids, labels) in enumerate(tqdm(train_loader,desc='Training',leave=False)):
                input_ids= input_ids.to(self.opt.device)
                labels= labels.to(self.opt.device)
                preds = model(input_ids)
                loss= criterion(preds, labels)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss)

            avg_loss = sum(epoch_loss)/len(epoch_loss)
            total_losses.append(avg_loss)
            print(f'Epoch {epoch+1}\t Training Loss: {avg_loss:.4f}')

            score=self._evaluate(model,val_loader,self.opt.device)
            
            if score >= score_record:
                score_record= score
                print('Update weight of model')
                best_params = model.state_dict()
                
        train_loss = sum(total_losses)/len(total_losses)
        torch.save(best_params,file_path)
        print(f'Total Loss: {train_loss:.4f}')
        
        return file_path
        
    def _evaluate(self,model, val_loader,device):
        with torch.no_grad():
            model.eval()
            for idx, (input_ids, labels) in enumerate(val_loader):
                input_ids= input_ids.to(device)
                labels= labels.to(device)
                
                preds = model(input_ids) # NxDxL
                preds = preds.argmax(1) # NxL
                
                preds = preds.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()
                try:
                    score= compute_metric(preds, labels)
                except:
                    score = f1_score(preds,labels,average='macro')
                
            print(f'Score: {score:.4f} ')
        return score
                
                


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_model', type=str)
    parser.add_argument('--pretrained_bert_name', default='distilbert/distilbert-base-uncased', type=str)
    parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--epochs', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=32, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--embedding_dim', default=256, type=int)
    parser.add_argument('--term_seq_len',default=100, type=int)
    parser.add_argument('--sentiment_seq_len',default=120,type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--num_layers',default=4,type=int)
    parser.add_argument('--device', default='cuda', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    
    
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

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
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.optimizer = optimizers[opt.optimizer]
    opt.term_num_classes=3
    
    if opt.device =='cuda':
        if torch.cuda.is_available():
            pass
        else:
            print('Cuda is not available!')
            torch.device='cpu'

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
                                    
        
        
        
        
    

