import torch
import torch.nn as nn


class Term_ABSAModel(nn.Module):
    def __init__(self,opt):
        super(Term_ABSAModel,self).__init__()
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.positional_encoding = nn.Embedding(opt.term_seq_len, opt.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=opt.embedding_dim, dim_feedforward=1024, dropout=0.2, nhead=opt.n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, opt.num_layers)
        self.conv1d = nn.Conv1d(opt.embedding_dim, 256, 3, 1,1)
        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64,3)
        )
        
    def forward(self, input_ids):
        positions= torch.arange(input_ids.size(1),device=input_ids.device).unsqueeze(0)
        embedded = self.embedding(input_ids)
        positional_encoding = self.positional_encoding(positions)
        inputs = embedded+ positional_encoding # NxLxD
        
        inputs = self.encoder(inputs)
        
        inputs = self.conv1d(inputs.permute(0,2,1)) # NxDxL
        inputs = inputs.permute(0,2,1)
        logits = self.fc(inputs) # NxLxD
        
        return logits.permute(0,2,1) #NxDxL 
    
    


class Sentiment_ABSAModel(nn.Module):
    def __init__(self,opt):
        super(Sentiment_ABSAModel,self).__init__()
        self.embedding = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.positional_encoding = nn.Embedding(opt.sentiment_seq_len, opt.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=opt.embedding_dim, dim_feedforward=1024, dropout=0.2, nhead=opt.n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, opt.num_layers)
        self.conv1d = nn.Conv1d(opt.embedding_dim, 256, 3, 1,1)
        
        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64,3)
        )
        
    def forward(self, input_ids):
        positions= torch.arange(input_ids.size(1),device=input_ids.device).unsqueeze(0)
        embedded = self.embedding(input_ids)
        positional_encoding = self.positional_encoding(positions)
        inputs = embedded+ positional_encoding # NxLxD
        
        inputs = self.encoder(inputs)
        
        inputs = self.conv1d(inputs.permute(0,2,1)) # NxDxL
        avg_pooling = nn.AvgPool1d(kernel_size=inputs.size(2))
        inputs= avg_pooling(inputs) #NxDx1
        inputs = inputs.squeeze(2) #NxD
        logits = self.fc(inputs) 
        
        return logits 