import pandas as pd
import matplotlib.pyplot as plt
import string
import csv
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from twitter_utils import utils, processing

SENT_THRESH = 0.2
WINDOW_SIZE = 5
USER_TWEET_THRESHOLD = 500
DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

FN_USERDATA = "data/raw/mil_bases/mil_base_user_level_data.csv"
FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
FN_OUT_RAW = "data/prepared/mil_base_user_level_00.csv"
FN_OUT_AGG = "data/prepared/mil_base_user_level_agg_00.csv"

DIR_USERS = "data/prepared/user_level_mb_old"

stopwords_eng = stopwords.words("english")
stopwords_eng += string.punctuation
stopwords_eng += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()
alx = utils.read_lexicon_to_dict(FN_LEXICON)

df = pd.read_csv(FN_USERDATA,
                 dtype={'fips': str},
                 parse_dates=['created_at'],
                 infer_datetime_format=True)
df = processing.preprocess_tweets_w_alex(df, tkz, ltz, stopwords_eng, alx, verbose=True, sent=stz)
df.to_csv(FN_OUT_RAW, index=False, quoting=csv.QUOTE_NONNUMERIC)
users = df['id'].unique()

# Create windowed data. First we group by user id and the date.
df['created_at'] = df['created_at'].apply(lambda dt: dt.date())
df['pos_sent'] = df['sentiment'].apply(lambda s: 1 if s > SENT_THRESH else 0)
df['neg_sent'] = df['sentiment'].apply(lambda s: 1 if s < -1 * SENT_THRESH else 0)
df['pos_anx'] = df['anxiety'].apply(lambda a: 1 if a > 0.0 else 0)
df['neg_anx'] = df['anxiety'].apply(lambda a: 1 if a < 0.0 else 0)
df_agg = df.groupby(['id', 'created_at']).agg(
    sum_anxiety=("anxiety", "sum"),
    count_pos_anx=("pos_anx", "sum"),
    count_neg_anx=("neg_anx", "sum"),
    ave_pos_anx=("pos_anx", "mean"),
    ave_neg_anx=("neg_anx", "mean"),
    sum_sentiment=("sentiment", "sum"),
    ave_sentiment=("sentiment", "mean"),
    count_pos_sent=("pos_sent", "sum"),
    count_neg_sent=("neg_sent", "sum"),
    ave_pos_sent=("pos_sent", "mean"),
    ave_neg_sent=("neg_sent", "mean"),
    total_tweets=("sentiment", "count")
)

num_saved = 0
for user, df_user in df_agg.groupby(level=0):

    if df_user['total_tweets'].sum() < USER_TWEET_THRESHOLD:
        continue

    df_user.reset_index(level=0, inplace=True)  # Remove the 0th level (user id)?
    df_user = df_user.reindex(DATE_RANGE, fill_value=0)
    df_windowed = df_user.rolling(window=WINDOW_SIZE).mean()
    df_windowed.to_csv("{}/{}_df.csv".format(DIR_USERS, user))
    num_saved += 1

print("users over threshold: {}".format(num_saved))


