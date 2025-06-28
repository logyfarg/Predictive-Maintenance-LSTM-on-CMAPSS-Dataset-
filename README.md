# Predictive Maintenance using LSTM

This project implements a Predictive Maintenance system using the NASA CMAPSS dataset. It trains an LSTM model to estimate the Remaining Useful Life (RUL) of jet engines based on sensor readings.

## Dataset
CMAPSS simulated jet engine sensor data. The data should be placed in:
data/CMAPSSData/

## Model
- Input: sequences of 30 cycles with 21 normalized sensor features.
- Architecture: LSTM + Dropout + Dense regression output.

## How to Run
Install dependencies:
pip install -r requirements.txt
Run training script:
python scripts/train_lstm.py

## Results
The model trains for 50 epochs and achieves progressively lower validation loss, showing it learns to predict RUL.

## Author
Logina Mahmoud

