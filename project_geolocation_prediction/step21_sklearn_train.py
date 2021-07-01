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

# TODO: download S3 files in your current data folder

# get_ipython().system('ls ./data')
data_dir = "../dataset/"


# In[2]:

def load_data(df_file):
    local_file = os.path.join(data_dir, df_file)
    df = pd.read_json(local_file)
    X, y = df['text'], df['label']
    return X, y

def extract_features(train_X, validation_X, test_X):
    # need to limit max features for memory concerns
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.2, max_features=100000)
    train_fea = vectorizer.fit_transform(train_X).astype('float32')
    validation_fea = vectorizer.transform(validation_X).astype('float32')
    test_fea = vectorizer.transform(test_X).astype('float32')
    vocab = vectorizer.vocabulary_
    return train_fea, validation_fea, test_fea, vocab


# In[ ]:

# Load prepared data
train_X, train_y = load_data("sagemaker_train.json")
validation_X, validation_y = load_data("sagemaker_validation.json")
test_X, test_y = load_data("sagemaker_test.json")
train_X, validation_X, test_X, vocab = extract_features(train_X, validation_X, test_X)


# In[ ]:
# Do GridSearch for hyper-parameter optimisation
# def gridSearch_rf():
#     turned_params = {
#         'n_estimators': [500, 10000],
#         'max_depth': [4, 6],
#         'n_jobs': [6],
#         'verbose': [1]
#     }
#     grid_clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=turned_params)
#     grid_clf.fit(train_X, train_y)
#     print(grid_clf.best_params_)

# Need to refit the model with the best params
# gridSearch_rf()


# In[ ]:

# define all benchmark classifiers
bayes = MultinomialNB(alpha=0.01)

lr = LogisticRegression(
    solver='saga',
    verbose=1,
    n_jobs=6,
    multi_class="multinomial",
    penalty='l1',
    max_iter=20,
    C=1
)

svc = LinearSVC(
    penalty='l1',
    tol=1e-3,
    multi_class="crammer_singer",
    dual=False,
    verbose=1
)


mlp = MLPClassifier(hidden_layer_sizes=(16, 16), random_state=2020, max_iter=20, solver='adam', tol=1e-3, verbose=True,
                    warm_start=True, early_stopping=True)

rf = RandomForestClassifier(n_estimators=100, max_depth=6, verbose=1, n_jobs=6, min_samples_split=5, min_samples_leaf=1)

clf_dict = {
    "MB": bayes,
    "LR": lr,
    "SVM": svc,
    "RF": rf,
    "MLP": mlp
}


# In[ ]:


# train model and persist them using joblib
for clf in clf_dict:
    s = datetime.now()
    print(f"Classifier {clf}")
    dump_file = '../modeltest/{}.joblib'.format(clf)
    if os.path.exists(dump_file):
        continue
    learner = clf_dict[clf]
    print(f"Classifier {clf} training ...")
    learner.fit(train_X, train_y)
    time_elapsed = datetime.now() - s
    print(f"Classifier {clf} trained in {time_elapsed}...")
    print(f"Classifier {clf} persisting ...")
    dump(learner, dump_file)




