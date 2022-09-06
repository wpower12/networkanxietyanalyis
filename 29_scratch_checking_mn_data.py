from naatools import utils

FN_RAW_DATA = "data/raw/mil_bases/historic_user_tweets_w_mn.csv"

mentioned_users = []
def collect_mn_users(s):
    global mentioned_users
    mentioned_users.extend(utils.str_to_list(s))


df = utils.load_huts_mn_df_from_fn(FN_RAW_DATA)
df['mentioned_users'].apply(collect_mn_users)
mentioned_users = set(mentioned_users)
tweet_user_set = set(df['user_id'].unique())

ids_2_process = mentioned_users-tweet_user_set

print(len(mentioned_users))
print(len(ids_2_process))
