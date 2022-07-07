import time
import pandas as pd
import snscrape.modules.twitter as sntwitter

DATE_RANGE = ['2022-01-02', '2022-06-20']
FN_USERS = "data/raw/mil_bases/mil_base_usernames.csv"
FN_OUT = "data/raw/mil_bases/historic_user_tweets.csv"
FN_PROC_USERS = "data/raw/mil_bases/processed_users.csv"

TIMEOUT_LEN_S = 15

user_df = pd.read_csv(FN_USERS)
proc_users = pd.read_csv(FN_PROC_USERS, header=None, dtype={0: int})[0].tolist()
n_users = len(user_df)
c = 0
for _, row in user_df.iterrows():
    u_id, _, u_name = row

    if u_id in proc_users:
        continue

    sns_query = "from:{} since:{} until:{}".format(u_name, DATE_RANGE[0], DATE_RANGE[1])
    successful = False
    user_tweets = []
    while not successful:
        try:
            user_tweets = []
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(sns_query).get_items()):
                user_tweets.append([tweet.date, u_id, tweet.id, tweet.content])
            successful = True
            user_df = pd.DataFrame(user_tweets, columns=['datetime', 'user_id', 'tweet_id', 'text'])
            user_df.to_csv(FN_OUT, mode='a', index=False)

            with open(FN_PROC_USERS, 'a') as f_pu:
                f_pu.write("{}\n".format(u_id))

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print("caught: {}".format(e))
            print("waiting and retrying")
            time.sleep(TIMEOUT_LEN_S)
    c += 1
    print("{:5d}/{} users".format(c, n_users))

