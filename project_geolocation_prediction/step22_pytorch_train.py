#!/usr/bin/env python
# coding: utf-8
# Importing the libraries needed
import os
from sklearn import preprocessing
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import re
import json
import logging
logging.basicConfig(level=logging.ERROR)


# In[2]:

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda:0' if cuda.is_available() else 'cpu'
assert device == "cuda:0"
print("*" * 60)
print(device)
print("*" * 60)

MAX_LEN = 512 # tweets may have up to 140 characters
BATCH_SIZE = 64
EPOCHS = int(os.environ.get("EPOCHS", 10))
print("EPOCHS: {}".format(EPOCHS))
LEARNING_RATE = 1e-05
TRAIN_BATCH_SIZE = 20
VALID_BATCH_SIZE = 4

data_dir = "../dataset/"


# In[3]:

def load_data(df_file):
    local_file = os.path.join(data_dir, df_file)
    df = pd.read_json(local_file)
    return df
train_df = load_data("sagemaker_train.json")
val_df = load_data("sagemaker_validation.json")
test_df = load_data("sagemaker_test.json")


# In[4]:

# map city names to indexes for deep learning labels
label_map = {}


def build_map(label_map, df):
    for _, label in df["label"].iteritems():
        if label not in label_map:
            label_map[label] = len(label_map)


build_map(label_map, train_df)
build_map(label_map, val_df)
build_map(label_map, test_df)
json.dump(label_map, open("../dataset/label_map.json", "w"))


# In[5]:

# update label column with indexes
def encode_labels(df):
    df['label'] = df['label'].apply(lambda x: label_map[x])
    return df['text'], df['label']


# In[6]:
train_X, train_y = encode_labels(train_df)
val_X, val_y = encode_labels(val_df)
test_X, test_y = encode_labels(test_df)

# In[7]:
# tokenizer = DistilBertTokenizer.from_pretrained(
#     'distilbert-base-uncased',
#     do_lower_case=True
# )
tokenizer = DistilBertTokenizer.from_pretrained(
    './distilbert-base-uncased',
    do_lower_case=True
)


# In[8]:
# Data triage objects
class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            #truncation=True, # either enable this or disable log warnings
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len


# In[9]:


training_set = Triage(train_df, tokenizer, MAX_LEN)
val_set = Triage(val_df, tokenizer, MAX_LEN)
testing_set = Triage(test_df, tokenizer, MAX_LEN)


# In[10]:

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# In[11]:
# The number of metro cities
class_num = 2993

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, class_num)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)  # Distilbert
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler) # append a dense layer
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)  # append a output layer
        return output


# In[12]:
# Fresh new training
model = DistillBERTClass()
model.to(device)


# In[13]:
# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()  # this loss is for multi-class
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# In[14]:
def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


# In[15]:
def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for idx, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        
        if idx%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print("Training Loss per 5000 steps: {}".format(loss_step))
            print("Training Accuracy per 5000 steps: {}".format(accu_step))

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print('The Total Accuracy for Epoch {}: {}'.format(epoch, (n_correct*100)/nb_tr_examples))
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print("Training Loss Epoch: {}".format(epoch_loss))
    print("Training Accuracy Epoch: {}".format(epoch_accu))

    return


# In[ ]:

# fresh training
for epoch in range(EPOCHS):
    print("Training {}".format(epoch))
    train(epoch)

print("Saving models for re-use")
model_file = '../modelfinal/pytorch_distilbert_{}.bin'.format(EPOCHS)
output_vocab_file = '../modelfinal/vocab_distilbert_{}.bin'.format(EPOCHS)
model_to_save = model
torch.save(model_to_save, model_file)
tokenizer.save_vocabulary(output_vocab_file)
print("Model saved")

