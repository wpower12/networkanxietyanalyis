import pandas as pd
import matplotlib.pyplot as plt
from twitter_utils import utils

DATE_RANGE = ['2022-01-02', '2022-06-20']
DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

FN_MB_DATA = "data/prepared/mil_base_df_2022_06_22.csv"
SENT_THRESHOLD = 0.25
WINDOW_SIZE = 5

OUT_DIR = "data/results/mil_base/2022_06_23_full_st_0.25"

df = pd.read_csv(FN_MB_DATA,
                 dtype={'fips': str},
                 parse_dates=['created_at'],
                 infer_datetime_format=True)
df['lemmas'] = df['lemmas'].apply(utils.str_to_list)

# Need to isolate over bases, and then days.
bases = sorted(df['idbase'].unique())
base_dfs = []
for baseid in bases:
    base_dfs.append(df[df['idbase'] == baseid].copy())
    # print(base_dfs[-1].head())

windowed_base_dfs = [
    ["fbragg", "Fort Bragg"],
    ["fcampbell", "Fort Campbell"],
    ["fhood", "Fort Hood"],
    ["jblmc", "Joint Base Lewis-McChord"],
    ["fbenning", "Fort Benning"],
    ["jbmdl", "Joint Base McGuire-Dix-Lakehurst"],
]

for bidx, bdf in enumerate(base_dfs):
    # print(bdf)
    # "idbase","basename","fips","created_at","text","sentiment","lemmas","anxiety"
    bdf['created_at'] = bdf['created_at'].apply(lambda dt: dt.date())
    bdf['pos_sent'] = bdf['sentiment'].apply(lambda s: 1 if s > SENT_THRESHOLD else 0)
    bdf['neg_sent'] = bdf['sentiment'].apply(lambda s: 1 if s < -1*SENT_THRESHOLD else 0)
    bdf['pos_anx'] = bdf['anxiety'].apply(lambda a: 1 if a > 0.0 else 0)
    bdf['neg_anx'] = bdf['anxiety'].apply(lambda a: 1 if a < 0.0 else 0)

    bdf_grouped = bdf.groupby(['idbase', 'created_at']).agg(
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

    bdf_grouped = bdf_grouped.reset_index()
    bdf_grouped.set_index('created_at', inplace=True)
    bdf_grouped = bdf_grouped.reindex(DATE_RANGE, fill_value=0)  # Fills in 'missing' dates
    bdf_grouped.drop(columns=["idbase"], inplace=True)
    bdf_windowed = bdf_grouped.rolling(window=WINDOW_SIZE).mean()
    windowed_base_dfs[bidx].append(bdf_windowed)
    bdf_windowed.to_csv("{}/{}_df.csv".format(OUT_DIR, windowed_base_dfs[bidx][0]))

# Make pretty pictures
for base_stub, base_name, base_df in windowed_base_dfs:
    # Sentiment
    base_df[['ave_pos_sent', 'ave_neg_sent', 'ave_sentiment']].plot()
    plt.title("{}".format(base_name))
    plt.savefig("{}/{}_sentiment.png".format(OUT_DIR, base_stub))

    # Anxiety
    base_df[['count_pos_anx', 'count_neg_anx']].plot()
    plt.title("{}".format(base_name))
    plt.savefig("{}/{}_anx.png".format(OUT_DIR, base_stub))

