import math

import pandas as pd
import matplotlib.pyplot as plt

FN_USERDATA = "data/raw/mil_bases/mil_base_user_level_data.csv"

df = pd.read_csv(FN_USERDATA,
                 dtype={'fips': str},
                 parse_dates=['created_at'],
                 infer_datetime_format=True)
df['created_at'] = df['created_at'].apply(lambda dt: dt.date())

print(len(df))
print(len(df['id'].unique()))
print(len(df['created_at'].unique()))

df_usergb = df.groupby(['id']).agg(
    num_tweets=('text', 'count')
)
df_usergb = df_usergb.reset_index()
df_usergb.set_index('id', inplace=True)
df_usergb['log_num_tweets'] = df_usergb['num_tweets'].apply(lambda n: math.log(n+1))

print(df_usergb.head())
print(df_usergb['num_tweets'].max())

print("threshold | users ")
for t in [5, 10, 50, 100, 200, 1000]:
    tdf = df_usergb[df_usergb['num_tweets'] > t]
    print("{:<5d} | {:<5d}".format(t, len(tdf)))

df_usergb.hist(column='num_tweets', bins=100, log=True)
plt.show()

