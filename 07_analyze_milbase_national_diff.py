import pandas as pd
import matplotlib.pyplot as plt

DIR_MIL_DATA = "data/results/mil_base/2022_06_23_full_st_0.25"
FN_NATIONAL  = "data/results/mil_base/national/national_windowed_df.csv"
DATE_RANGE   = ['2022-02-12', '2022-03-10']

# Note - This is windowed data. So each day is an aggreagate over the preceeding X days (X=5 here)
bases = [
    ["fbragg", "Fort Bragg"],
    ["fcampbell", "Fort Campbell"],
    ["fhood", "Fort Hood"],
    ["jblmc", "Joint Base Lewis-McChord"],
    ["fbenning", "Fort Benning"],
    ["jbmdl", "Joint Base McGuire-Dix-Lakehurst"],
]

df_nat = pd.read_csv(FN_NATIONAL, parse_dates=[0], index_col=0)
for base_stub, base_name in bases:
    df = pd.read_csv("{}/{}_df.csv".format(DIR_MIL_DATA, base_stub), parse_dates=[0], index_col=0)
    df = df[DATE_RANGE[0]:DATE_RANGE[1]]

    diff_df = (df-df_nat).copy()
    diff_df[['ave_pos_sent', 'ave_neg_sent', 'ave_sentiment']].plot()
    plt.title("{}".format(base_name))
    plt.savefig("{}/{}_sentiment_natdiff.png".format(DIR_MIL_DATA, base_stub))
