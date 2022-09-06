import pandas as pd
from naatools import gather

DATE_RANGE = ['2021-11-11', '2022-04-11']
STATES = ['idaho', 'montana', 'oregon']
DATA_DIR = "data/rafaa"

for state in STATES:
    fn_user_ids = "{}/users_{}.csv".format(DATA_DIR, state)
    fn_proc_ids = "{}/proc_uids_{}.csv".format(DATA_DIR, state)
    fn_out = "{}/tweets_{}.csv".format(DATA_DIR, state)

    gather.collect_huts(fn_user_ids, fn_proc_ids, fn_out, DATE_RANGE)
