import time
import pandas as pd
import csv
from tqdm import tqdm
import snscrape.modules.twitter as sntwitter

DATE_RANGE = ['2022-01-02', '2022-06-20']
FN_USERS = "data/raw/mil_bases/mil_base_usernames.csv"
FN_OUT = "data/raw/mil_bases/historic_user_tweets_w_mn.csv"
FN_PROC_USERS = "data/raw/mil_bases/processed_users_mn.csv"

TIMEOUT_LEN_S = 15
MAX_ATTEMPTS = 1

# All features are numbers (integers, even)
FEATURES = ["u_acct_age",
            "u_n_followers",
            "u_n_following",
            "u_n_favorites",
            "u_n_lists",
            "u_n_tweets",
            "t_n_retweets",
            "t_n_hashtags",
            "t_n_user_mentions",
            "t_n_urls",
            "t_n_chars",
            "t_n_digits"]
DF_COLUMNS = ['datetime', 'user_id', 'tweet_id', 'text', 'mentioned_user_ids']
DF_COLUMNS.extend(FEATURES)


def count_digits(text):
    n_digits = 0
    for c in text:
        if c in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            n_digits += 1
    return n_digits


def get_tweet_features(tweet):
    user = tweet.user
    u_ca = user.created
    t_ca = tweet.date

    u_acct_age = (t_ca-u_ca).days  # bc datetime? is it that easy? yup. wow.
    u_n_followers = user.followersCount if user.followersCount else 0
    u_n_following = user.friendsCount if user.friendsCount else 0
    u_n_favorites = user.favouritesCount if user.favouritesCount else 0
    u_n_lists = user.listedCount if user.listedCount else 0
    u_n_tweets = user.statusesCount if user.statusesCount else 0
    t_n_retweets = tweet.retweetCount
    t_n_hashtags = len(tweet.hashtags) if tweet.hashtags else 0
    t_n_user_mentions = len(tweet.mentionedUsers) if tweet.mentionedUsers else 0
    t_n_urls = len(tweet.outlinks) if tweet.outlinks else 0
    t_n_chars = len(tweet.content)
    t_n_digits = count_digits(tweet.content)
    return [u_acct_age, u_n_followers, u_n_following, u_n_favorites, u_n_lists, u_n_tweets,
            t_n_retweets, t_n_hashtags, t_n_user_mentions, t_n_urls, t_n_chars, t_n_digits]


user_df = pd.read_csv(FN_USERS)
proc_users = pd.read_csv(FN_PROC_USERS, header=None, dtype={0: int})[0].tolist()

n_users = len(user_df)
c_users = 0
for _, row in tqdm(user_df.iterrows(), total=n_users):
    u_id, _, u_name = row

    if u_id in proc_users:
        continue

    sns_query = "from:{} since:{} until:{}".format(u_name, DATE_RANGE[0], DATE_RANGE[1])
    successful = False
    attempts = 0
    while not successful and attempts < MAX_ATTEMPTS:
        try:
            user_tweets = []
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(sns_query).get_items()):
                features = get_tweet_features(tweet)
                if tweet.mentionedUsers is not None:
                    mentions = [u.id for u in tweet.mentionedUsers]
                else:
                    mentions = []
                row = [tweet.date, u_id, tweet.id, tweet.content, mentions]
                row.extend(features)
                user_tweets.append(row)

            successful = True
            user_df = pd.DataFrame(user_tweets, columns=DF_COLUMNS)
            user_df.to_csv(FN_OUT, mode='a', index=False, quoting=csv.QUOTE_NONNUMERIC, header=False)

            with open(FN_PROC_USERS, 'a') as f_pu:
                f_pu.write("{}\n".format(u_id))

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print("caught: {}".format(e))
            print("waiting and retrying")
            time.sleep(TIMEOUT_LEN_S)
            attempts += 1

