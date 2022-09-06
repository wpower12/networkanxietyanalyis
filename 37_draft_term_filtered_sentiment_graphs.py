import pandas as pd
import matplotlib.pyplot as plt
import csv
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from naatools import utils, processing
import spam

SEED = 4
SENT_THRESH = 0.2
AFIN_THRESH = 1
WINDOW_SIZE = 5
DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

# FN_HUTS = "data/raw/mil_bases/historic_user_tweets_cleaned.csv"
FN_HUTS = "data/mb_raw_mentioned_tweets_POS_EXAMPLES.csv"

FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
FN_SH_ARFF = "data/raw/spam/95k-continuous.arff"
FN_OUT_RAW = "data/raw/mil_bases/huts_preprocessed.csv"
FN_AFINN_LEX = "data/raw/AFINN-111.txt"

OUT_DIR = "data/results/mil_base/2022_09_05_milterm_features_mentioned_users_00"
if not os.path.exists(OUT_DIR): os.mkdir(OUT_DIR)

# 'Military' filter terms.
FILTER_TERMS = ["military", "army", "navy", "airforce", "marines", "Military", "Army", "Navy", "AirForce", "Marines"]

# Log the parameters used for this run.
with open("{}/exp_context.txt".format(OUT_DIR), 'w') as f:
    f.write("exp script: 37_draft_term_filtered_sentiment_graphs.py")
    f.write("results dir: {}\n".format(OUT_DIR))
    f.write("seed: {}\n".format(SEED))
    f.write("sentiment_thresh: {}\n".format(SENT_THRESH))
    f.write("sentiment_afinn_thresh: {}\n".format(AFIN_THRESH))
    f.write("window_size: {}\n".format(WINDOW_SIZE))
    f.write("milterms used:\n")
    f.write(", ".join(FILTER_TERMS))

# Spam Filtering ###
print("loading and applying spam filter")
df_huts = pd.read_csv(FN_HUTS,
                      names=spam.util.FULL_HUTS_COL_HEADERS,
                      parse_dates=["datetime"],
                      infer_datetime_format=True,
                      low_memory=False)
sh_clf  = spam.models.load_spam_ham_pipeline(FN_SH_ARFF)  # Trains an SH classifier based on the provided spam data.
X_huts  = spam.util.huts_tensor_from_df(df_huts)
df_huts['spam'] = sh_clf.predict(X_huts)
df_huts = df_huts[df_huts['spam'] == 0]

# Preprocessing ###
print("loading nlp components")
tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()
alx = utils.read_lexicon_to_dict(FN_LEXICON)
afn = utils.read_afinn(FN_AFINN_LEX)

utils.set_random_seeds(SEED)

print("processing data")
df_huts = processing.preprocess_tweets_w_alex(df_huts, tkz, ltz, utils.stopwords_list(), alx, verbose=True, sent=stz, afinn=afn)

# Term Filtering ###
print("creating term-filtered dataset")
df_mil_terms = processing.filter_terms(df_huts, FILTER_TERMS)
print("size of filtered mil-term dataset: {}".format(len(df_mil_terms)))


def create_rolling_sequences(df):
    df['datetime'] = df['datetime'].apply(lambda dt: dt.date())
    df['pos_sent'] = df['sentiment'].apply(lambda s: 1 if s > SENT_THRESH else 0)
    df['neg_sent'] = df['sentiment'].apply(lambda s: 1 if s < -1 * SENT_THRESH else 0)

    df['pos_afinn'] = df['sentiment_afinn'].apply(lambda s: 1 if s > AFIN_THRESH else 0)
    df['neg_afinn'] = df['sentiment_afinn'].apply(lambda s: 1 if s < -1 * AFIN_THRESH else 0)

    df['pos_anx'] = df['anxiety'].apply(lambda a: 1 if a > 0.0 else 0)
    df['neg_anx'] = df['anxiety'].apply(lambda a: 1 if a < 0.0 else 0)

    df_agg = df.groupby(['datetime']).agg(
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
        sum_sentiment_afinn=("sentiment_afinn", "sum"),
        ave_sentiment_afinn=("sentiment_afinn", "mean"),
        count_pos_afinn=("pos_afinn", "sum"),
        count_neg_afinn=("neg_afinn", "sum"),
        ave_pos_afinn=("pos_afinn", "mean"),
        ave_neg_afinn=("neg_afinn", "mean"),
        total_tweets=("sentiment", "count")
    )

    df_agg = df_agg.reset_index()
    df_agg.set_index('datetime', inplace=True)
    df_agg = df_agg.reindex(DATE_RANGE, fill_value=0)  # Fills in 'missing' dates
    return df_agg.rolling(window=WINDOW_SIZE).mean()


# Sequence Generation ###
print("generating sequences of windowed data.")
df_full_sequences = create_rolling_sequences(df_huts)
df_milt_sequences = create_rolling_sequences(df_mil_terms)


def save_fig(df, feature_list, label):
    df[feature_list].plot()
    plt.title("{} Sentiment".format(label))
    plt.savefig("{}/{}.png".format(OUT_DIR, label))


print("saving figures and csvs.")
features_VADER = ['ave_pos_sent', 'ave_neg_sent', 'ave_sentiment']
save_fig(df_full_sequences, features_VADER, "VADER Sentiment - All Tweets")
save_fig(df_milt_sequences, features_VADER, "VADER Sentiment - Mil Tweets")

features_AFINN = ['ave_pos_afinn', 'ave_neg_afinn', 'ave_sentiment_afinn']
save_fig(df_full_sequences, features_AFINN, "AFINN Sentiment - All Tweets")
save_fig(df_milt_sequences, features_AFINN, "AFINN Sentiment - Mil Tweets")

df_full_sequences.to_csv("{}/{}.csv".format(OUT_DIR, "all_tweets"))
df_milt_sequences.to_csv("{}/{}.csv".format(OUT_DIR, "mil_term_tweets"))
