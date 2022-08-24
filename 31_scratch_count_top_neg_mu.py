import tweepy
from tqdm import tqdm
from decouple import config
from twitter_utils import utils

FN_RAW_DATA = "data/raw/mil_bases/historic_user_tweets_w_mn.csv"
TOP_N_MUS = 40000
OUT_FN = "data/raw/mil_bases/top_mus.csv"


mentioned_users = dict()
def count_freq_mn_users(s):
    global mentioned_users
    mus = utils.str_to_list(s)
    for mu in mus:
        if mu in mentioned_users:
            mentioned_users[mu] += 1
        else:
            mentioned_users[mu] = 1


print("calculating mu frequencies.")
df = utils.load_huts_mn_df_from_fn(FN_RAW_DATA)
df['mentioned_users'].apply(count_freq_mn_users)
mentioned_users = [(k, mentioned_users[k]) for k in mentioned_users]
mentioned_users = sorted(mentioned_users, key=lambda k: -1*k[1])
print(len(mentioned_users))
top_N_ids = [k[0] for k in mentioned_users[:TOP_N_MUS]]


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


auth = tweepy.OAuthHandler(config('T_CONSUME_KEY'), config('T_CONSUME_SECRET'))
auth.set_access_token(config('T_ACCESS_KEY'), config('T_ACCESS_SECRET'))
api = tweepy.API(auth)

id_batches = list(chunks(top_N_ids, 99))
print("finding user names for top N user ids.")
rows = []
for id_batch in tqdm(id_batches, unit="user batch"):
    try:
        # user = api.get_user(user_id=u_id)
        # rows.append("{}, {}, {}\n".format(u_id, user.name, user.screen_name))
        users = api.lookup_users(user_id=id_batch)
        for user in users:
            rows.append("{}, nvm, {}\n".format(user.id, user.screen_name))
    except KeyboardInterrupt as kbi:
        raise kbi
    except Exception as e:
        # raise e
        pass

print("writing to file")
with open(OUT_FN, 'w') as f:
    f.write("id,name,screen_name\n")
    for r in rows:
        f.write(r)