import pandas as pd

FN_LEXICON = "data/srikar/anxiety_lexicon.csv"
OUT_FN = "data/prepared/anxiety_lexicon_filtered.csv"

df = pd.read_csv(FN_LEXICON, header=0)

p = df['anxiety'] > 0.8
n = df['anxiety'] < -0.8
df = df[p | n]

df.to_csv(OUT_FN)
