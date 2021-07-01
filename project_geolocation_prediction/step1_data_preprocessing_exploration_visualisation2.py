#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#
# # Create local data dir
# get_ipython().run_line_magic('mkdir', '../data')
#
#
# # In[4]:
#
#
# # Install dependencies
# import sys
# # https://github.com/erikavaris/tokenizer
# get_ipython().system('{sys.executable} -m pip install git+https://github.com/erikavaris/tokenizer.git')
# # https://pypi.org/project/stop-words/
# get_ipython().system('{sys.executable} -m pip install stop-words')



# In[32]:


# Download data from S3 to local data dir
# import boto3
import os
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline


from tokenizer import tokenizer as TT
twk = TT.TweetTokenizer()
from stop_words import get_stop_words
stopwords = set(get_stop_words('en'))

# s3 = boto3.client('s3')
# BUCKET_NAME = 'geopred'
# s3.download_file(BUCKET_NAME, 'sagemaker_validation.jl', '../dataset/sagemaker_validation.jl')
#s3.download_file('geopred', 'sagemaker_test.jl', '../data/sagemaker_test.jl')
#s3.download_file('geopred', 'sagemaker_train.jl', '../data/sagemaker_train.jl')


# In[22]:

local_folder = "../dataset/"


def preprocess(text_list):
    """ All preprocessings for tweets: lowercase, concatenation, twitter tokenisation and stopwords removal
    """
    processed = []
    text_list = text_list[0] if isinstance(text_list, pd.Series) else text_list
    for text in text_list:
        lower = text.lower()
        tokens = twk.tokenize(lower)
        cleaned = [tok for tok in tokens if tok not in stopwords]
        processed.append(" ".join(cleaned))
    # join tweets by new line or return None for further dropna
    processed = "\n".join(processed) if processed else None
    return processed


def persist_data(df, data_type):
    """ Upload transformed data to S3
    """
    df_file = f"sagemaker_{data_type}.json"
    local_file = os.path.join(local_folder, df_file)
    df.to_json(local_file)
    # with open(local_file, "rb") as f:
    #     s3.upload_fileobj(f, BUCKET_NAME, df_file)
    #     print(f"Upload is successful for {df_file}")
    print(f"success for {df_file}")


def viz(df, data_type):
    """ Show visualisation figures"""
    # plot city distribution and top 10 cities
    cities = df['label'].value_counts().sort_values(ascending=False)
    print(f"Top 10 populated cities in {data_type}")
    print(cities[:10])
    print(f"\n\nData description")
    print(cities.describe())

    plt.rcParams.update({'font.size': 14})

    fig, axs = plt.subplots(4, 1, figsize=(15, 18))
    fig.tight_layout(pad=3.0)
    cities.plot.box(ax=axs[1], showfliers=False,
                    title=f"City Boxplot Distributions: {data_type}")

    cities.plot.hist(bins=100, ax=axs[0], title=f"City Distributions: {data_type}")

    features = df[['text']].apply(lambda r: len(r[0].split()), axis=1)
    features.plot.box(ax=axs[3], showfliers=False,
                      title=f"Feature Boxplot Distributions: {data_type}")

    features.plot.hist(bins=100, ax=axs[2], title=f"Feature Distributions: {data_type}")
    plt.show()


def etl(data_type):
    """ETL interface for integrated loading, cleansing and visualisation"""
    assert data_type in ("validation", "train", "test"), f"Incorrect data type: {data_type}"
    print(f"******** Analysing {data_type} ********\n\n")
    # download dataset to local dir if not cached
    local_file = os.path.join(local_folder, f'sagemaker_{data_type}.jl')
    # if not os.path.exists(local_file):
        # s3.download_file(BUCKET_NAME,
        #                  f'sagemaker_{data_type}.jl',
        #                  local_file)

    # transform, persist data if not cached
    transformed_file = os.path.join(local_folder, f"sagemaker1_{data_type}.json")
    if not os.path.exists(transformed_file):
        # load into pandas
        df = pd.read_json(local_file, lines=True)

        # report basic statistics
        print(f"Before processing")
        print(f"Training data shape: {df.shape}\n")
        print(f"Examples peak:\n {df.head()}\n\n")

        # apply col-wise transformation for preprocessings
        df[['text']] = df[['text']].apply(lambda r: preprocess(r), axis=1)

        # drop empty rows in case all user's tweets are stop words
        df = df.dropna(axis=0)
        print(f"After processing")
        print(f"Training data shape: {df.shape}\n")
        print(f"Examples peak:\n {df.head()}\n\n")

        # Persist ML ready to use data
        persist_data(df, data_type)
    else:
        df = pd.read_json(transformed_file)
    # show viz
    viz(df, data_type)


etl("train")
etl("validation")
etl("test")