import torch
import torch.nn as nn


class Term_BertModel(nn.Module):
    def __init__(self,bert, opt):
        super(Term_BertModel,self).__init__()
        self.backbone = nn.Sequential(*list(bert.children()))[:-2]
        self.fc = nn.Sequential(
            nn.Linear(768,512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256,3)
        )

    def forward(self, input):
        input = self.backbone(input).last_hidden_state
        logits = self.fc(input)
        
        return logits.permute(0,2,1)
    
    


class Sentiment_BertModel(nn.Module):
    def __init__(self, bert, opt):
        super(Sentiment_BertModel,self).__init__()
        self.backbone = nn.Sequential(*list(bert.children()))[:-2]
        self.fc = nn.Sequential(
            nn.Linear(768,512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256,3)
        )

    def forward(self, input):
        input = self.backbone(input).last_hidden_state[:,0,:].squeeze(1)
        logits = self.fc(input)
        return logits
        



