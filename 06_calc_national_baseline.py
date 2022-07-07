import pandas as pd
import csv
import matplotlib.pyplot as plt
from twitter_utils import utils

DATE_RANGE = ['2022-02-12', '2022-03-10']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

FN_NATIONAL_DATA = "data/prepared/national_df_20220212_20220310.csv"
SENT_THRESHOLD = 0.25
WINDOW_SIZE = 5

OUT_DIR = "data/results/mil_base/national"

df = pd.read_csv(FN_NATIONAL_DATA,
                 dtype={'fips': str},
                 parse_dates=['created_at'],
                 infer_datetime_format=True)
df['lemmas'] = df['lemmas'].apply(utils.str_to_lemma_list)

df['created_at'] = df['created_at'].apply(lambda dt: dt.date())
df['pos_sent'] = df['sentiment'].apply(lambda s: 1 if s > SENT_THRESHOLD else 0)
df['neg_sent'] = df['sentiment'].apply(lambda s: 1 if s < -1*SENT_THRESHOLD else 0)
df['pos_anx'] = df['anxiety'].apply(lambda a: 1 if a > 0.0 else 0)
df['neg_anx'] = df['anxiety'].apply(lambda a: 1 if a < 0.0 else 0)

df_grouped = df.groupby(['created_at']).agg(
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

df_grouped = df_grouped.reset_index()
df_grouped.set_index('created_at', inplace=True)
df_grouped = df_grouped.reindex(DATE_RANGE, fill_value=0)  # Fills in 'missing' dates
df_windowed = df_grouped.rolling(window=WINDOW_SIZE).mean()

# Make pretty pictures
df_windowed[['ave_pos_sent', 'ave_neg_sent', 'ave_sentiment']].plot()
plt.title("{}".format("National - sentiment"))
plt.savefig("{}/{}_sentiment.png".format(OUT_DIR, "national"))

# Anxiety
df_windowed[['count_pos_anx', 'count_neg_anx']].plot()
plt.title("{}".format("National - anxiety"))
plt.savefig("{}/{}_anx.png".format(OUT_DIR, "national"))

df_windowed.to_csv("{}/national_windowed_df.csv".format(OUT_DIR), quoting=csv.QUOTE_NONNUMERIC)
