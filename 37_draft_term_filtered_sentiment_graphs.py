import pandas as pd
import matplotlib.pyplot as plt
import csv
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from twitter_utils import utils, processing
import spam

SENT_THRESH = 0.2
WINDOW_SIZE = 5
DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

# TODO - Change this to point to the last known good raw data.
FN_HUTS = "data/raw/mil_bases/historic_user_tweets_cleaned.csv"

FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
FN_SH_ARFF = "data/raw/spam/95k-continuous.arff"
FN_OUT_RAW = "data/raw/mil_bases/huts_preprocessed.csv"

OUT_DIR = "data/results/mil_base/2022_09_05_milterm_features"

# 'Military' filter terms.
FILTER_TERMS = ["military", "army", "navy", "airforce", "marines", "Military", "Army", "Navy", "AirForce", "Marines"]


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
df_huts = processing.preprocess_tweets_w_alex(df_huts, tkz, ltz, utils.stopwords_list(), alx, verbose=True, sent=stz)
df_huts.to_csv(FN_OUT_RAW, index=False, quoting=csv.QUOTE_NONNUMERIC)


# Term Filtering ###
print("creating term-filtered dataset")
df_mil_terms = processing.filter_terms(df_huts, FILTER_TERMS)
print("size of filtered mil-term dataset: {}".format(len(df_mil_terms)))


# Sequence Generation ###
print("generating sequences of windowed data.")
df_full_sequences = processing.create_rolling_sequences(df_huts, SENT_THRESH, DATE_RANGE, WINDOW_SIZE)
df_milt_sequences = processing.create_rolling_sequences(df_mil_terms, SENT_THRESH, DATE_RANGE, WINDOW_SIZE)


def save_fig(df, label):
    df[['ave_pos_sent', 'ave_neg_sent', 'ave_sentiment']].plot()
    plt.title("{} Sentiment".format(label))
    plt.savefig("{}/{}.png".format(OUT_DIR, label))


print("saving figures and csvs.")
save_fig(df_full_sequences, "Mentioned Users - All Tweets")
save_fig(df_milt_sequences, "Mentioned Users - Mil Tweets")

df_full_sequences.to_csv("{}/{}.csv".format(OUT_DIR, "all_tweets"))
df_milt_sequences.to_csv("{}/{}.csv".format(OUT_DIR, "mil_term_tweets"))
