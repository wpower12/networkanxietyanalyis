import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "data/results/sus_vs_mn_00"
LOG_COLS = ["type", "batch", "value"]

gsm_log = pd.read_csv("{}/{}".format(RESULTS_DIR, "gsm_log.csv"), names=LOG_COLS)
susm_log = pd.read_csv("{}/{}".format(RESULTS_DIR, "susm_log.csv"), names=LOG_COLS)


def get_series_from_df(df, start, stop):
    df = df[start:stop]
    batch_training_loss = df[df['type'] == 'batch_training_loss']
    batch_test_loss = df[df['type'] == 'batch_test_loss']
    x_train, y_train = batch_training_loss['batch'], batch_training_loss['value']
    x_test, y_test = batch_test_loss['batch'], batch_test_loss['value']
    return x_train, y_train, x_test, y_test


# (91227, 182454), (182454, 273681)
x_gsm_training, y_gsm_training, x_gsm_test, y_gsm_test = get_series_from_df(gsm_log, 182454, 273681)
x_susm_training, y_susm_training, x_susm_test, y_susm_test = get_series_from_df(susm_log, 182454, 273681)

fig, ax = plt.subplots()

ax.plot(x_gsm_training, y_gsm_training, label="GSM Model")
ax.plot(x_susm_training, y_susm_training, label="SUS Model")
ax.legend()
plt.show()
