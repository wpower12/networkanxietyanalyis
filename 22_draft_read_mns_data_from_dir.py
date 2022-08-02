from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm
from twitter_utils import processing

USER_SEQ_DIR = "data/prepared/mb_user_sequences"
WINDOW_SIZE = 5
ANX_THRESHOLD = 0.1
# DATE_RANGE = ['2022-01-02', '2022-06-20']
# DATE_RANGE = pd.date_range(start=DATE_RANGE[0], end=DATE_RANGE[1], freq="D")

user_files = [f for f in listdir(USER_SEQ_DIR) if isfile(join(USER_SEQ_DIR, f))]

print("reading raw user sequences.")
# Need to build the user_id -> df map.
user_id_2_df = {}
for f in tqdm(user_files, delay=0.1):
    user_id = int(f.replace("sequences_", "").replace(".csv", ""))
    user_df = pd.read_csv("{}/{}".format(USER_SEQ_DIR, f),
                          parse_dates=[0],
                          infer_datetime_format=True,
                          dtype={"mentioned_users": str})
    user_df['mentioned_users'].fillna("", inplace=True)
    user_id_2_df[user_id] = user_df

print("building example graph sequences")
example_graphs = []
example_labels = []
for _, user_id in enumerate(tqdm(user_id_2_df, delay=0.1)):
    df_central_user = user_id_2_df[user_id]
    graphs, labels = processing.create_examples_from_raw_mn_sequences(df_central_user,
                                                                      user_id,
                                                                      user_id_2_df,
                                                                      WINDOW_SIZE,
                                                                      ANX_THRESHOLD)
    example_graphs.extend(graphs)
    example_labels.extend(labels)

print(example_graphs[0], example_labels[0])
