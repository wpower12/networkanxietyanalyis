from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
from tqdm import tqdm
from twitter_utils import processing

WINDOW_SIZE = 7
ANX_THRESHOLD = 0.0

DIR_USER_SEQS = "data/results/mil_base/user_sequences"
FN_SAVE_X = "data/prepared/mil_base/X.npy"
FN_SAVE_y = "data/prepared/mil_base/y.npy"

user_files = [f for f in listdir(DIR_USER_SEQS) if isfile(join(DIR_USER_SEQS, f))]
# user_ids = [int(f.replace("sequences_", "").replace(".csv", "")) for f in user_files]

Xs, ys = [], []
for fn_user in tqdm(user_files):
    df_user = pd.read_csv("{}/{}".format(DIR_USER_SEQS, fn_user),
                          parse_dates=[0],
                          infer_datetime_format=True)
    user_X, user_y = processing.create_examples_from_full_sequences(df_user, WINDOW_SIZE, ANX_THRESHOLD)
    Xs.append(user_X)
    ys.append(user_y)

X = np.concatenate(Xs)
y = np.concatenate(ys)
print(X.shape, y.shape)
np.save(FN_SAVE_X, X)
np.save(FN_SAVE_y, y)
