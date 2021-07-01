# Importing the libraries needed
import os
import json
from sklearn import preprocessing
from datetime import datetime
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
from haversine import haversine
import re
# To disable Bert input truncation warnings
import logging
logging.basicConfig(level=logging.ERROR)

from torch import cuda
# gpus = torch.cuda.device_count()
# print(gpus)
# GPUID = int(os.environ.get("GPUID", 0))
# device = 'cuda' if cuda.is_available() else 'cpu'
# if device == "cuda":
#     device = 'cuda:{}'.format(GPUID)
# assert device.startswith("cuda")
device = 'cuda:0' if cuda.is_available() else 'cpu'
assert device == "cuda:0"
print("*" * 60)
print(device)
print("*" * 60)

# experiment settings

data_dir = "../dataset/"
MAX_LEN = 512 # tweets may have up to 140 characters
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("EPOCHS", 10))
print("Training with EPOCHS: {}".format(EPOCHS))
LEARNING_RATE = 1e-05
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 10


# Data load function
def load_data(df_file):
    local_file = os.path.join(data_dir, df_file)
    df = pd.read_json(local_file)
    return df

test_df = load_data("sagemaker_test.json")
val_df = load_data("sagemaker_validation.json")

# Make maps from city name (used by evaluation) to prediction labels (used by Bert)
label_map_file = "../dataset/label_map.json"
assert os.path.exists(label_map_file), "Missing label map file for evaluation"
label_map = json.load(open(label_map_file))


# Transform city labels to encoded labels
def encode_labels(df):
    df['label'] = df['label'].apply(lambda x: label_map[x])
    return df['text'], df['label']


test_X, test_y = encode_labels(test_df)
val_X, val_y = encode_labels(val_df)


# Bert input tokeniser
# tokenizer = DistilBertTokenizer.from_pretrained(
#     'distilbert-base-uncased',
#     do_lower_case=True
# )
tokenizer = DistilBertTokenizer.from_pretrained(
    './distilbert-base-uncased',
    do_lower_case=True
)


class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, class_num)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# Dataset object
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
            #truncation=True,
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


testing_set = Triage(test_df, tokenizer, MAX_LEN)
val_set = Triage(val_df, tokenizer, MAX_LEN)
test_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }
testing_loader = DataLoader(testing_set, **test_params)
val_loader = DataLoader(val_set, **test_params)


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


def valid(model, val_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    results = []
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)
            results.append((big_idx, targets))

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if _%5000==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print("Validation Loss Epoch: {}".format(epoch_loss))
    print("Validation Accuracy Epoch: {}".format(epoch_accu))
    
    return epoch_accu, results


id2city = {v: k for k, v in label_map.items()}
city2coords = {}
with open("../dataset/city2coords.jl") as fr:
    for l in fr:
        city, lat, lon = json.loads(l)
        city2coords[city] = (lat, lon)


# evaluation by my customisation
def eval_distance(results):
    acc = 0.0
    dlist = []
    datalist = []
    for (a, b) in results:
        a = torch.Tensor.cpu(a).detach().numpy()
        b = torch.Tensor.cpu(b).detach().numpy()
        for ai, bi in zip(a, b):
            if ai == bi:
                acc += 1
            ca = id2city[ai]
            cb = id2city[bi]
            d = haversine(city2coords[ca], city2coords[cb])
            datalist.append([[city2coords[ca][0], city2coords[ca][1]], ca, [city2coords[cb][0], city2coords[cb][1]],
                             cb, d])
            dlist.append(d)
    acc = round(acc/(VALID_BATCH_SIZE * len(results)), 4)
    dlist.sort()
    mean = sum(dlist) / (VALID_BATCH_SIZE * len(results))
    median = dlist[int(len(dlist)/2)]
    return acc, round(median), round(mean), datalist


model_file = "../modeltest/pytorch_distilbert_{}.bin".format(EPOCHS)
model = torch.load(model_file, map_location='cuda:0')

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

acc, results = valid(model, val_loader)
print("Accuracy on test data = %0.2f%%" % acc)

acc, median, mean, datalist = eval_distance(results)
print("DistilBert acc: {}, median: {}, mean: {}".format(acc, median, mean))

data_file = "../modeltest/datalist.json"
with open(data_file, 'w') as f:
    json.dump({'datalist': datalist}, f)




