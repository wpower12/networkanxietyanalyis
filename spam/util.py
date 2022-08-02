import pandas as pd
import numpy as np
import arff

ARFF_COL_HEADERS = ["account_age",
                    "no_follower",
                    "no_following",
                    "no_userfavourites",
                    "no_lists",
                    "no_tweets",
                    "no_retweets",
                    "no_hashtag",
                    "no_usermention",
                    "no_urls",
                    "no_char",
                    "no_digits",
                    "tweet_class"]

HUTS_COL_HEADERS = ["u_acct_age",
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


def load_tweet_df_from_arff(fn):
    rows = []
    for d in arff.load(fn):
        rows.append(list(d))
    df = pd.DataFrame(rows, columns=ARFF_COL_HEADERS)
    df['spam'] = df['tweet_class'].apply(lambda t: 1 if t == 'spammer' else 0)
    df.drop(columns=['tweet_class'], inplace=True)
    return df


def load_tweet_tensors_from_arff(fn):
    df = load_tweet_df_from_arff(fn)
    return np.hsplit(df.to_numpy(), np.array([12]))


def load_huts_df_from_csv(fn):
    df_huts = pd.read_csv(fn,
                          header=0,
                          parse_dates=["datetime"],
                          infer_datetime_format=True,
                          low_memory=False)
    # df_huts = df_huts[df_huts['user_id'] != 'user_id']
    return df_huts


def load_huts_tensor_from_csv(fn):
    df_huts = load_huts_df_from_csv(fn)
    df_huts = df_huts[HUTS_COL_HEADERS]
    df_huts = df_huts.astype('int')
    return df_huts.to_numpy()


def huts_tensor_from_df(df_huts):
    df_huts = df_huts[HUTS_COL_HEADERS]
    df_huts = df_huts.astype('int')
    return df_huts.to_numpy()
