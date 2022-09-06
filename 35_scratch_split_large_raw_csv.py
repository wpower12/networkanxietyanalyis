import pandas as pd
from naatools import utils

FN_INPUT_SMALL = "data/raw/mb_raw_mentioned_tweets_POS_EXAMPLES.csv"
FN_INPUT_LARGE = "data/mb_raw_mentioned_tweets_POS_EXAMPLES.csv"

df = utils.load_huts_mn_df_from_fn(FN_INPUT_LARGE)

print(len(df))
