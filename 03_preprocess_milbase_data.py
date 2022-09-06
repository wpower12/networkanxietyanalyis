import pandas as pd
from naatools import processing as proc

import string
import csv
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

SENT_THRESH = 0.2

FN_MB_DATA = "data/raw/mil_bases/full_milbase_data_2022_06_22.csv"
FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
FN_OUT = "data/prepared/mil_base_df_2022_06_22.csv"

stopwords_eng = stopwords.words("english")
stopwords_eng += string.punctuation
stopwords_eng += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()

df = pd.read_csv(FN_MB_DATA, header=0, dtype={'fips': str})

# before we can call the new method, we need to turn the anxiety lexicon into a dict.
def read_lexicon_to_dict(fn):
    lex_dict = {}
    with open(fn, 'r') as f:
        f.readline()
        for l in f.readlines():
            _, lemma, _, score = l.split(",")
            score = float(score)
            lex_dict[lemma] = score
    return lex_dict


alex = read_lexicon_to_dict(FN_LEXICON)

df = proc.preprocess_tweets_w_alex(df, tkz, ltz, stopwords_eng, alex, verbose=True, sent=stz)

print(df.head())

n_0  = len(df[df['anxiety'] == 0])
n_n0 = len(df[df['anxiety'] != 0])
print("{} non zero out of {} anxiety scores.".format(n_n0, n_0 + n_n0))

n_gtt = len(df[df['sentiment'] > SENT_THRESH])
n_ltt = len(df[df['sentiment'] < -1*SENT_THRESH])
print("sentiments: {} gtt, {} ltt".format(n_gtt, n_ltt))

df.to_csv(FN_OUT, index=False, quoting=csv.QUOTE_NONNUMERIC)  # Note - We need to start doing that as a best practice.
