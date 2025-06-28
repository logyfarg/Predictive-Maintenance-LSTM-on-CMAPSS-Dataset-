import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Paths
data_dir = "data/CMAPSSData"
train_file = os.path.join(data_dir, "train_FD001.txt")

# Load data
cols = ["unit_number", "time_in_cycles"] + \
       [f"op_setting_{i}" for i in range(1, 4)] + \
       [f"sensor_measurement_{i}" for i in range(1, 22)]

train_df = pd.read_csv(train_file, sep=" ", header=None)
train_df = train_df.loc[:, ~train_df.columns.str.match('Unnamed')]
train_df.columns = cols

# Compute RUL
rul_df = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
rul_df.columns = ["unit_number", "max_cycle"]
train_df = train_df.merge(rul_df, on="unit_number", how="left")
train_df["RUL"] = train_df["max_cycle"] - train_df["time_in_cycles"]
train_df.drop("max_cycle", axis=1, inplace=True)

# Normalize
scaler = MinMaxScaler()
feature_cols = train_df.columns.difference(["unit_number", "time_in_cycles", "RUL"])
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# Sequence preparation
sequence_length = 30
sequences = []
labels = []

for unit in train_df["unit_number"].unique():
    unit_data = train_df[train_df["unit_number"] == unit]
    features = unit_data.drop(["unit_number", "time_in_cycles", "RUL"], axis=1).values
    rul = unit_data["RUL"].values
    for i in range(len(features) - sequence_length):
        sequences.append(features[i:i+sequence_length])
        labels.append(rul[i+sequence_length])

X_train = np.array(sequences)
y_train = np.array(labels)

# Model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# Train
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
