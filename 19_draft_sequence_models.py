import numpy as np

from keras import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import RMSprop
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

TRAIN_FRAC = 0.9
VAL_FRAC = 0.1

FN_SAVE_X = "data/prepared/mil_base/X.npy"
FN_SAVE_y = "data/prepared/mil_base/y.npy"

X = np.load(FN_SAVE_X)
y = np.load(FN_SAVE_y)

# Remove NaNs
X_o = X[:, 0, 0]
nan_flags = np.isnan(X_o)
X = X[~nan_flags]
y = y[~nan_flags]
print(X.shape, y.shape)

# TODO - shuffle.

# Split test train
tt_split_id = int(TRAIN_FRAC*len(X))
X_train, y_train = X[:tt_split_id], y[:tt_split_id]
X_test,  y_test  = X[tt_split_id:], y[tt_split_id:]
# Split validation out of training
tv_split_id = int(tt_split_id*VAL_FRAC)
X_val, y_val = X_train[-tv_split_id:], y_train[-tv_split_id:]
X_train, y_train = X_train[:-tv_split_id], y_train[:-tv_split_id]

print(X_val.shape, y_val.shape)

model = Sequential([
    SimpleRNN(2),
    Dense(1)
])

model.compile(
    optimizer=RMSprop(),
    loss=BinaryCrossentropy(),
    metrics=[BinaryAccuracy()]
)

history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=2,
    validation_data=(X_val, y_val)
)

results = model.evaluate(X_test, y_test, batch_size=64)
print("test loss, acc: ", results)


