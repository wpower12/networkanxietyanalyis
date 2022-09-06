import pandas as pd
from os import listdir

from tqdm import tqdm

import tweepy
from decouple import config

import snscrape.modules.twitter as sntwitter

from naatools import utils

# iterate over the directory of user sequences. Each csv file is a 'complete' (with 0's) sequence of user data.
# for each user, we look for the indexs with 'positive' labels. That is, the field 'max_raw_anxiety' is 1 (or -1?)
# then we look at the preceeding W days and gather their mentioned users. These are in the mentioned_users field
# and are a string because you hate yourself.

DATE_RANGE = ['2022-01-02', '2022-06-20']
OUT_FN = "data/prepared/mb_mentioned_user_ids.csv"
NEW_OUT_FN = "data/prepared/mb_mentioned_users_RIGHTCOLS.csv"
DIR_USER_SEQS = "data/prepared/mb_user_sequences_new_targets"
WINDOW_SIZE = 5

# mentioned_user_ids = set()
# for fn in listdir(DIR_USER_SEQS):
#     df = pd.read_csv("{}/{}".format(DIR_USER_SEQS, fn))
#     mentioned_users = []
#     pos_labels = df[df['max_raw_anx'] == 1]
#     for pos_row in pos_labels.iterrows():
#         loc = pos_row[0]
#         for w in range(WINDOW_SIZE):
#             past_idx = loc-w-1
#             if past_idx >= 0:
#                 past_mus = df.iloc[past_idx]['mentioned_users']
#                 past_mus = "{}".format(past_mus)
#                 if past_mus == "\"\"" or past_mus == "nan":
#                     continue
#                 past_mus = past_mus[:-1].split(",")
#                 mentioned_users.extend(past_mus)
#
#     mentioned_user_ids = mentioned_user_ids.union(set(mentioned_users))
#
# print(len(mentioned_user_ids))
#
# with open(OUT_FN, 'w') as f:
#     for mu in list(mentioned_user_ids):
#         f.write("{}\n".format(mu))

auth = tweepy.OAuthHandler(config('T_CONSUME_KEY'), config('T_CONSUME_SECRET'))
auth.set_access_token(config('T_ACCESS_KEY'), config('T_ACCESS_SECRET'))
api = tweepy.API(auth)

ids = []
with open(OUT_FN, 'r') as f:
    ids = [l.replace("\n", "") for l in f.readlines()]
    ids = ids[:-1]
    ids = [int(i) for i in ids]

# from: https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


id_batches = list(chunks(ids, 99))
print("finding user names.")
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
with open(NEW_OUT_FN, 'w') as f:
    f.write("id,name,screen_name\n")
    for r in rows:
        f.write(r)
