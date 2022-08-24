import pandas as pd

FN_POS_MUS = "data/prepared/mb_mentioned_users_RIGHTCOLS.csv"
FN_NEG_MUS = "data/raw/mil_bases/top_mus.csv"

FN_OUT = "data/prepared/mb_neg_mentioned_users.csv"

df_pos = pd.read_csv(FN_POS_MUS)
df_neg = pd.read_csv(FN_NEG_MUS)

users_pos = set(df_pos['id'].unique())
users_neg = set(df_neg['id'].unique())

print(len(users_neg))
print(len(users_neg))
users_to_collect = list(users_neg.difference(users_pos))
print(len(users_to_collect))

df_to_collect = df_neg[df_neg.isin(users_to_collect)]
print(len(df_to_collect))

df_to_collect.to_csv(FN_OUT)
