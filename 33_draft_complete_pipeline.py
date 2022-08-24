import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from twitter_utils import pipeline, utils

SENT_THRESH = 0.2
ANX_THRESHOLD = 0.1
WINDOW_SIZE = 5
DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

### Source Files ###
FN_RAW_CU_TWEETS = "data/raw/mil_bases/historic_user_tweets_w_mn.csv"
FN_RAW_MU_TWEETS = "data/prepared/mb_raw_mentioned_tweets.csv"

### Outputs
FN_PROCESSED_CU_TWEETS = "data/prepared/mb_processed_central_tweets.csv"
FN_PROCESSED_MU_TWEETS = "data/prepared/mb_processed_mentioned_tweets.csv"
DIR_CENTRAL_USERS      = "data/prepared/mb_00_central_users"
DIR_MENTIONED_USERS    = "data/prepared/mb_00_mentioned_users"
#######################################################################################################################

print("## initializing NLP components.")
FN_LEXICON = "data/prepared/anxiety_lexicon_filtered.csv"
tkz = TweetTokenizer(strip_handles=True, reduce_len=True)
ltz = WordNetLemmatizer()
stz = SentimentIntensityAnalyzer()
alx = utils.read_lexicon_to_dict(FN_LEXICON)

print("## preprocessing central users.")
pipeline.process_raw_tweets_to_user_dfs(FN_RAW_CU_TWEETS,
                                        tkz,
                                        ltz,
                                        alx,
                                        stz,
                                        SENT_THRESH,
                                        DATE_RANGE,
                                        FN_PROCESSED_CU_TWEETS,
                                        DIR_CENTRAL_USERS,
                                        process_mn=True)

print("## preprocessing mentioned users.")
pipeline.process_raw_tweets_to_user_dfs(FN_RAW_MU_TWEETS,
                                        tkz,
                                        ltz,
                                        alx,
                                        stz,
                                        SENT_THRESH,
                                        DATE_RANGE,
                                        FN_PROCESSED_MU_TWEETS,
                                        DIR_MENTIONED_USERS,
                                        process_mn=False)

# print("## generating examples")
# pipeline.generate_exs_from_cu_mu_dirs(DIR_CENTRAL_USERS, DIR_MENTIONED_USERS, WINDOW_SIZE, ANX_THRESHOLD)

