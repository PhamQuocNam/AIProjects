import torch
import torch.nn as nn
from utils.preprocessing import build_tokenizer, build_corpus, bert_tokenizer,\
     text_normalize, pad_and_truncate
from Models.base_model import Term_ABSAModel,Sentiment_ABSAModel
from Models.bert import Term_BertModel,Sentiment_BertModel
from transformers import AutoModelForTokenClassification



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
        text= text_normalize(text)
        print('Normalize :', text)
        if 'bert' in self.opt.model_name:
            input_ids= self.tokenizer.tokenize(text)
            input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
            temp=input_ids
            input_ids = pad_and_truncate(input_ids,self.opt.term_seq_len,self.tokenizer.convert_tokens_to_ids('<padding>'))
            input_ids = torch.tensor(input_ids,device= self.opt.device).unsqueeze(0)
            logits = self.term_model(input_ids)
            logits = logits.argmax(1).squeeze(0)
            aspect_term =[]
            for i in range(len(temp)):
                if logits[i] != 0:
                    aspect_term.append(self.tokenizer.convert_ids_to_tokens(temp[i]))
            aspect_term= ' '.join(aspect_term)
            input = self.tokenizer(text,aspect_term)['input_ids']
            input = pad_and_truncate(input,self.opt.sentiment_seq_len, self.tokenizer.convert_tokens_to_ids('<padding>'))
            input= torch.tensor(input,device=self.opt.device).unsqueeze(0)
            output= self.sentiment_model(input).argmax(1)[0]
            
            if output ==0:
                output='neural'
            elif output==1:
                output='positive'
            else:
                output='negative'
            
            print(f'Aspect term: {aspect_term}, Polarity: {output}')
        else:
            input_ids = [self.tokenizer.token_to_id(token) if self.tokenizer.token_to_id(token) else 0 for token in text.split(' ')]
            input_ids = pad_and_truncate(input_ids, self.opt.term_seq_len, self.tokenizer.token_to_id('<padding>'))
            temp= input_ids
            input_ids = torch.tensor(input_ids, device=self.opt.device).unsqueeze(0)
            logits= self.term_model(input_ids)
            logits = logits.argmax(1).squeeze(0)
            aspect_term=[]
            
            for i in range(len(temp)):
                
                if logits[i] !=0:
                    aspect_term.append(self.tokenizer.id_to_token(temp[i]))
            aspect_term= ' '.join(aspect_term)
            input = text+ ' <sep> ' +aspect_term
            input= self.tokenizer.token_to_id(input)
            input= pad_and_truncate(input, self.opt.sentiment_seq_len, self.tokenizer.token_to_id('<padding>'))
            input= torch.tensor(input,device=self.opt.device).unsqueeze(0)
            output= self.sentiment_model(input).argmax(1)[0]
            
            if output ==0:
                output='neural'
            elif output==1:
                output='positive'
            else:
                output='negative'
            
            print(f'Aspect term: {aspect_term}, Polarity: {output}')


if __name__ == '__main__':
    model_classes = {
        'base_model': [Term_ABSAModel,Sentiment_ABSAModel],
        'bert_model': [Term_BertModel,Sentiment_BertModel]
    }
    
    class Option(object): pass
    opt = Option()
    opt.model_name = 'bert_model'
    opt.model_class = model_classes[opt.model_name]
    
    # set your trained models here
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.term_seq_len = 100
    opt.sentiment_seq_len=120
    opt.bert_dim = 768
    opt.pretrained_bert_name = 'distilbert/distilbert-base-uncased'
    opt.polarities_dim = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = inference(opt)
    inf.evaluate('this laptop run so fast')
                                         
        
        
        
        
        
        