import pandas as pd
import os
import csv
from tqdm import tqdm
from naatools import processing, utils
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DIR_INPUT_FILES = "data/raw/mil_bases/split_huts_00"
DIR_USER_TWEET_FILES = "data/prepared/processed_mus_00_tweets"
DIR_USER_AGGED_FILES = "data/prepared/processed_mus_00_agged"

print("--initializing nlp components.")
FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()
alx = utils.read_lexicon_to_dict(FN_LEXICON)

print("processing split raw tweet files from {}".format(DIR_INPUT_FILES))
for fn in os.listdir(DIR_INPUT_FILES):
    df_raw = utils.load_huts_mn_df_from_fn("{}/{}".format(DIR_INPUT_FILES, fn))
    df_raw = processing.preprocess_tweets_w_alex(df_raw,
                                                 tkz,
                                                 ltz,
                                                 utils.stopwords_list(),
                                                 alx,
                                                 verbose=True,
                                                 sent=stz,
                                                 mn=False)  # Only True for central users. don't need it for mentioned.

    users = df_raw['user_id'].unique()
    for u, user in enumerate(tqdm(users, delay=0.25, unit=" user", leave=False)):
        df_user = df_raw[df_raw['user_id'] == user]
        fn_user = "{}/user_{}.csv".format(DIR_USER_TWEET_FILES, user)
        df_user.to_csv(fn_user, index=False, quoting=csv.QUOTE_NONNUMERIC, mode='a')

# print("aggregating user data.")
# for fn in os.listdir(DIR_USER_TWEET_FILES):
#     df_user_seq_raw = create_user_df_w_aggregated_mn(df_user, sent_thresh, date_range)
#     df_user_seq_raw.to_csv("{}/sequences_{}.csv".format(dir_save_users, user))