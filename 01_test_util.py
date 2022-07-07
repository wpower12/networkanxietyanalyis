from twitter_utils import processing as proc
from twitter_utils import utils
import pandas as pd
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DATA_FN_STUB = "2022_01_01_to_07_thresh_{}.csv"
DATA_DIR = "data/raw/thresholded"
OUT_FN_STUB = "2022010107_v00_{}"
OUT_DIR = "data/prepared/thresholded"

ANXIETY_KW_FN = "data/prepared/anxiety_lexicon_filtered.csv"

THRESHOLDS = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

stopwords_eng = stopwords.words("english")
stopwords_eng += string.punctuation
stopwords_eng += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()

df_anxkws = pd.read_csv(ANXIETY_KW_FN, header=0, index_col=0)
anxiety_kws = list(df_anxkws['lemma'].values)
anxiety_dict = utils.anx_dict_from_df(df_anxkws)

for threshold in THRESHOLDS:
    # The first 3 files are created by the preprocess_tweets2bow method.
    fn_lemmas = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format("{}_{}.csv".format(threshold, "lemmas")))
    fn_bow    = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format("{}_{}.csv".format(threshold, "bow")))
    fn_dict   = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format("{}_{}.csv".format(threshold, "dict")))
    fn_topics = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format("{}_{}.csv".format(threshold, "topics")))
    fn_pp_df  = "{}/{}".format(OUT_DIR, OUT_FN_STUB.format("{}_{}.csv".format(threshold, "pp_df")))  # The processed DF.

    df = pd.read_csv("{}/{}".format(DATA_DIR, DATA_FN_STUB.format(threshold)),
                     engine="python",
                     header=0,
                     index_col=0,
                     dtype={"userid": str, "text": str})

    # 1 - applies preprocessing to tokenize, lemmatize, and sentiment-analyze the raw tweet text.
    #     returns a data frame with the text replaced by lemmas (and sent scores based on raw text, if stz provided)
    df, text_dict = proc.preprocess_tweets(df,
                                           tkz,
                                           ltz,
                                           stopwords_eng,
                                           OUT_DIR,
                                           OUT_FN_STUB.format(threshold),
                                           verbose=True,
                                           sent=stz)  # Will calculate sentiment scores.

    df_bow = pd.read_csv(fn_bow, dtype={"userid": str})
    text_dict = utils.read_text_dict(fn_dict)

    tts = proc.generate_topic_terms(df_bow, text_dict, fn_topics, verbose=True)
    keeper_terms = list(tts)
    keeper_terms.extend(anxiety_kws)

    df_lemmas = pd.read_csv(fn_lemmas, index_col=0, header=0, dtype={"userid": str})
    df_lemmas_clean, text_dict_clean = proc.filter_lemmas(df_lemmas, keeper_terms)

    proc.add_anxiety_scores(df_lemmas_clean, anxiety_dict)

    n_0 = len(df_lemmas_clean[df_lemmas_clean['anxiety'] == 0])
    n_n0 = len(df_lemmas_clean[df_lemmas_clean['anxiety'] != 0])
    print("{} non zero out of {} anxiety scores.".format(n_n0, n_0 + n_n0))

    df_lemmas_clean.to_csv(fn_pp_df)
    print("saved cleaned df to {}".format(fn_pp_df))
