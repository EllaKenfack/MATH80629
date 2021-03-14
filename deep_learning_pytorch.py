# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:57:04 2021

@author: Admin
"""
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

# We import home functions to split data
import sys
sys.path.append('C:/Users/Admin/Documents/Cours/MATH80629/Project/Analysis')
from utilities import split_stratified_into_train_val_test

import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.optim as optim

import spacy
import time

#import en_core_web_sm

"""
Importing and Analyzing the Dataset

"""
seed=12345

#Import raw file
filepath=r'C:\Users\Admin\Documents\Cours\MATH80629\Project\Analysis'
filename =filepath+'\dataset.txt'
dataset = pd.read_csv(filename,sep=",")
print(dataset.head(5))

#Select relevant feature
data_worry = dataset.loc[:, ['worry','text_long']]

#split stratify into train-valid-test
df_train, df_val, df_test = split_stratified_into_train_val_test(data_worry, stratify_colname='worry', frac_train=0.60, frac_val=0.20, frac_test=0.20,random_state=seed)

#save to json file
df_train.to_json (filepath+'\\train.json', orient='records',lines=True)
df_val.to_json (filepath+'\\valid.json', orient='records',lines=True)
df_test.to_json  (filepath+'\\test.json', orient='records',lines=True)


# #see an example of the dataset
# data_worry.dtypes      
# print(data_worry.head(5))
# data_worry['text_long'][3]

#see the distribution of target variable
#df_test['worry'].value_counts()
# plt.hist(data_worry['worry'], bins=9)

torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)

fields = {'worry': ('label', LABEL), 'text_long': ('text', TEXT)}


train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = filepath,
                                        train = 'train.json',
                                        validation = 'valid.json',
                                        test = 'test.json',
                                        format = 'json',
                                        fields = fields)

print(vars(train_data[0]))

"""
Preparing the Embedding Layer

"""

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)



BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    device = device)

 #sort_within_batch = False !!!!!!!!!!!!!!!  TO BE VERIFIED
"""

Recurrent Neural Network
--define and instantiate the model

"""

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden)
    

#Instantiating the model

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM, 
            EMBEDDING_DIM, 
            HIDDEN_DIM, 
            OUTPUT_DIM, 
            N_LAYERS, 
            BIDIRECTIONAL, 
            DROPOUT, 
            PAD_IDX)

# Counting the number of parameters

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


#We then replace the initial weights of the embedding layer with the pre-trained embeddings.
pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

"""

Recurrent Neural Network
--Instatntiating the training

"""       

#Define the optimizer    
optimizer = optim.Adam(model.parameters())

#Define the loss function
criterion = nn.L1Loss()

model = model.to(device)
criterion = criterion.to(device)

#metric function
def MAE(preds, y):
    mae=torch.abs(preds - y).mean() 
    return mae


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_mae = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        text, text_lengths = batch.text
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        mae = MAE(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_mae += mae.item()
        
    return epoch_loss / len(iterator), epoch_mae / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_mae = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            mae = MAE(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_mae += mae.item()
        
    return epoch_loss / len(iterator), epoch_mae / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    

"""

Recurrent Neural Network
--Training the model

"""  

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train MAE: {train_acc:.2f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. MAE: {valid_acc:.2f}')

