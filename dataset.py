import torch
import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# SentencePiece 모델 로드
model_file = "./data/kowiki.model" 
sp = spm.SentencePieceProcessor()
sp.load(model_file)

class ChatBotDataset(Dataset):
    def __init__(self, csv_path, max_seq_len, train=True):
        super(ChatBotDataset, self).__init__()
        self.csv_path = csv_path
        self.train = train
        self.df = pd.read_csv(csv_path)
        
        self.Q = self.df['Q'].values.tolist()
        self.A = self.df['A'].values.tolist()
        
        self.x = torch.tensor([sp.EncodeAsIds(q) + (max_seq_len-len(sp.EncodeAsIds(q)))*[0] for q in self.Q], dtype=torch.long)
        self.y = torch.tensor([sp.EncodeAsIds(a) + (max_seq_len-len(sp.EncodeAsIds(a)))*[0] for a in self.A], dtype=torch.long)
                
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.x, self.y, test_size=0.1, random_state=42)
        
    def __getitem__(self, index):
        
        if self.train:
            return self.train_x[index], self.train_y[index]
        else:
            return self.val_x[index], self.val_y[index]

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.val_x)
        