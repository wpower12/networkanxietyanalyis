import pandas as pd
from tqdm import tqdm
from naatools import utils, processing
import os

FN_HUTS_MN = "data/raw/mil_bases/historic_user_tweets_w_mn.csv"
USER_OUT_DIR = "data/prepared/mb_raw_user_tweets_by_day_00"

if not os.path.exists(USER_OUT_DIR):
    os.mkdir(USER_OUT_DIR)

df_huts = utils.load_huts_mn_df_from_fn(FN_HUTS_MN)
df_huts['datetime'] = df_huts['datetime'].apply(lambda dt: dt.date())

# User Sequences
users = df_huts['user_id'].unique()

most_tweets = 0
most_single_day = 0

print("splitting users", flush=True)
for u, user in enumerate(tqdm(users, delay=0.25, unit=" user")):
    df_user = df_huts[df_huts['user_id'] == user].copy()
    df_user_by_day = df_user.groupby(['datetime'])

    user_dir_path = "{}/user_{}".format(USER_OUT_DIR, user)
    if not os.path.exists(user_dir_path):
        os.mkdir(user_dir_path)

    for name, group in df_user_by_day:
        group.to_csv("{}/day_{}.csv".format(user_dir_path, name))

    df_agg = df_user_by_day.agg(
        daily_tweets=("user_id", "count")
    )

    if len(df_user) > most_tweets:
        most_tweets = len(df_user)

    if df_agg['daily_tweets'].max() > most_single_day:
        most_single_day = df_agg['daily_tweets'].max()

print("{} users, most tweets: {}, most single day: {}".format(len(users), most_tweets, most_single_day))
