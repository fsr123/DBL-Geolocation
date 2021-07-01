#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load
from haversine import haversine
from datetime import datetime

# get_ipython().system('ls ./data')
data_dir = "../dataset/"


# In[2]:
# build city2coord dict for distance-based calculation
# e.g., city1: (lat1, lon1) vs city2: (lat2, lon2)
import json

city2coords = {}
with open("../dataset/city2coords.jl") as fr:
    for l in fr:
        city, lat, lon = json.loads(l)
        city2coords[city] = (lat, lon)


# In[ ]:


def load_data(df_file):
    local_file = os.path.join(data_dir, df_file)
    df = pd.read_json(local_file)
    X, y = df['text'], df['label']
    return X, y


def extract_features(train_X, validation_X, test_X):
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.2, max_features=100000)
    train_fea = vectorizer.fit_transform(train_X).astype('float32')
    validation_fea = vectorizer.transform(validation_X).astype('float32')
    test_fea = vectorizer.transform(test_X).astype('float32')
    vocab = vectorizer.vocabulary_
    return train_fea, validation_fea, test_fea, vocab


# Load prepared data
train_X, train_y = load_data("sagemaker_train.json")
validation_X, validation_y = load_data("sagemaker_validation.json")
test_X, test_y = load_data("sagemaker_test.json")
_, _, test_X, _ = extract_features(train_X, validation_X, test_X)


# In[ ]:


def eval_test(clf, test_X, test_y):
    pred_y = clf.predict(test_X)
    test_y_ = [o for _, o in test_y.iteritems()]
    assert len(pred_y) == len(test_y_) > 1
    acc = 0.0
    dlist = []
    for a, b in zip(pred_y, test_y_):
        if a == b:
            acc += 1
        d = haversine(city2coords[a], city2coords[b])
        dlist.append(d)
    acc = round(acc / len(pred_y), 4)
    dlist.sort()
    mean = sum(dlist) / len(pred_y)
    median = dlist[int(len(dlist)/2)]
    return acc, round(median), round(mean)


# In[ ]:


clf_list = ["MB", "LR", "SVM", "RF", "MLP"]
for clf in clf_list:
    model_file = '../modeltest/{}.joblib'.format(clf)
    print(f"Loading models from {model_file}")
    model = load(model_file)
    print(f"{clf} loaded")
    acc, median, mean = eval_test(model, test_X, test_y)
    print(f"{clf}: {acc}, {median}, {mean}")    


# In[ ]:




