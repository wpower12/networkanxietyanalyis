import pandas as pd
import numpy as np
import pickle
import csv
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from naatools import utils, processing
import spam

SENT_THRESH = 0.2
WINDOW_SIZE = 5
ANX_THRESHOLD = 0.1
DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

# FN_HUTS = "data/raw/mil_bases/historic_user_tweets_cleaned.csv"
FN_HUTS_MN = "data/raw/mil_bases/historic_user_tweets_w_mn.csv"
FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
FN_SH_ARFF = "data/raw/spam/95k-continuous.arff"

FN_OUT_RAW = "data/prepared/mil_base/huts_mn_preprocessed_bert_features.csv"
HUTS_OUT_DIR = "data/prepared/mil_base/"
USER_OUT_DIR = "data/prepared/mb_user_sequences_bert_features"

# Spam Filtering ###
# print("loading and applying spam filter")
# sh_clf  = spam.models.load_spam_ham_pipeline(FN_SH_ARFF)  # Trains an SH classifier based on the provided spam data.
df_huts = utils.load_huts_mn_df_from_fn(FN_HUTS_MN)
# X_huts  = spam.util.huts_tensor_from_df(df_huts)
# df_huts['spam'] = sh_clf.predict(X_huts)

# print("size before spam filter: {}".format(len(df_huts)))
# df_huts = df_huts[df_huts['spam'] == 0]
# print("size after after filter: {}".format(len(df_huts)))

df_huts = df_huts[:1000]

# Preprocessing ###
print("preprocessing spam filtered dataset")
tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()
alx = utils.read_lexicon_to_dict(FN_LEXICON)
# BERT embedding model and tokenizer.

bert_m = AutoModel.from_pretrained("vinai/bertweet-base")
bert_t = AutoTokenizer.from_pretrained("vinai/bertweet-base")

df_huts = processing.preprocess_tweets_w_alex(df_huts,
                                              tkz,
                                              ltz,
                                              utils.stopwords_list(),
                                              alx,
                                              verbose=True,
                                              sent=stz,
                                              mn=True,
                                              bert_model=bert_m,
                                              bert_token=bert_t)
df_huts.to_csv(FN_OUT_RAW, index=False, quoting=csv.QUOTE_NONNUMERIC)

# User Sequences
users = df_huts['user_id'].unique()
total_events = 0
total_examples = 0

raw_seq_dfs = {}

# First pass - Raw Sequences for each user - NOT ROLLED UP -> cant get that to work rn.
print("generating", flush=True)
for u, user in enumerate(tqdm(users, delay=0.25, unit=" user")):
    df_user = df_huts[df_huts['user_id'] == user].copy()
    df_user_seq_raw = processing.create_user_df_w_aggregated_mn(df_user, SENT_THRESH, DATE_RANGE)
    df_user_seq_raw.to_csv("{}/sequences_{}.csv".format(USER_OUT_DIR, user))
    raw_seq_dfs[user] = df_user_seq_raw  # These are NOT rolled up yet, so each day is just that days info.
