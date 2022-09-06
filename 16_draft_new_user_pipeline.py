import pandas as pd
import numpy as np
import csv
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from naatools import utils, processing
import spam

SENT_THRESH = 0.2
WINDOW_SIZE = 5
ANX_THRESHOLD = 0.1
DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

FN_HUTS = "data/raw/mil_bases/historic_user_tweets_cleaned.csv"
FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
FN_SH_ARFF = "data/raw/spam/95k-continuous.arff"
FN_OUT_RAW = "data/raw/mil_bases/huts_preprocessed.csv"
HUTS_OUT_DIR = "data/results/mil_base/huts"
USER_OUT_DIR = "data/results/mil_base/user_sequences"


# Spam Filtering ###
print("loading and applying spam filter")
sh_clf  = spam.models.load_spam_ham_pipeline(FN_SH_ARFF)  # Trains an SH classifier based on the provided spam data.
df_huts = spam.util.load_huts_df_from_csv(FN_HUTS)
X_huts  = spam.util.huts_tensor_from_df(df_huts)
df_huts['spam'] = sh_clf.predict(X_huts)

print("size before spam filter: {}".format(len(df_huts)))
df_huts = df_huts[df_huts['spam'] == 0]
print("size after after filter: {}".format(len(df_huts)))

# Preprocessing ###
print("preprocessing spam filtered dataset")
tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()
alx = utils.read_lexicon_to_dict(FN_LEXICON)
df_huts = processing.preprocess_tweets_w_alex(df_huts, tkz, ltz, utils.stopwords_list(), alx, verbose=False, sent=stz)
df_huts.to_csv(FN_OUT_RAW, index=False, quoting=csv.QUOTE_NONNUMERIC)

# User Sequences
users = df_huts['user_id'].unique()
total_events = 0
total_examples = 0
for u, user in enumerate(users):
    df_user = df_huts[df_huts['user_id'] == user].copy()
    df_user_sequences = processing.create_rolling_sequences(df_user, SENT_THRESH, DATE_RANGE, WINDOW_SIZE)
    df_user_sequences.to_csv("{}/sequences_{}.csv".format(USER_OUT_DIR, user))
    X, y = processing.create_examples_from_full_sequences(df_user_sequences, WINDOW_SIZE, ANX_THRESHOLD)
    total_events += np.sum(y)
    total_examples += len(X)
    print("processed user {:5d}/{} | {:5d}/{:5d} events".format(u+1, len(users), total_events, total_examples))



