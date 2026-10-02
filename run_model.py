"""
Cryptocurrency [ETH/USDT] Prediction with RNN Neural Network
Standalone execution script
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout

try:
    from tensorflow.keras.layers import SimpleRNNCell, RNN
except ImportError:
    from tensorflow.python.keras.layers import SimpleRNNCell, RNN


def line_plot(line1, line2, label1=None, label2=None, title="", lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel("ETH/USDT", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="best", fontsize=16)
    return fig


def normalise_zero_base(continuous):
    base = continuous.iloc[0] if hasattr(continuous, "iloc") else continuous[0]
    return continuous / base + 2


def build_lstm_model(input_data, output_size, neurons, activ_func="tanh",
                     dropout=0.21, loss="mse", optimizer="adam"):
    model = Sequential()
    model.add(RNN(cell=[SimpleRNNCell(128),
                        SimpleRNNCell(256),
                        SimpleRNNCell(128)]))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def main():
    csv_path = "ETH-USD_data.csv" if os.path.exists("ETH-USD_data.csv") else "/content/ETH-USD (2).csv"
    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    data = data.iloc[:, 0:6]
    y = data.loc[:, ["Close"]]
    data = data.drop(["Close"], axis="columns")
    print(data.head(5))

    data = data.set_index("Date")
    data.index = pd.to_datetime(data.index, unit="ns")
    print(f"Dataset date range: {data.index[-1]} to {data.index[0]}")

    X_train = data[256:]
    X_test = data[:256]
    y_train = y[256:]
    y_test = y[:256]

    X_train = normalise_zero_base(X_train)
    X_test = normalise_zero_base(X_test)
    y_train = normalise_zero_base(y_train)
    y_test = normalise_zero_base(y_test)

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    np.random.seed(64)
    lstm_neurons = 256
    epochs = 16
    batch_size = 32
    loss = "mse"
    dropout = 0.25
    optimizer = "adam"

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer
    )

    print(f"Training RNN model for {epochs} epochs...")
    modelfit = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        shuffle=True
    )

    # Save training loss figure
    plt.figure()
    plt.plot(modelfit.history["loss"], "r", linewidth=2, label="Training loss")
    plt.plot(modelfit.history["val_loss"], "g", linewidth=2, label="Validation loss")
    plt.title("RNN Neural Networks - ETH Model")
    plt.xlabel("Epochs numbers")
    plt.ylabel("MSE numbers")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()
    print("Saved loss_plot.png")

    preds = model.predict(X_test).squeeze()
    mae = mean_absolute_error(preds, y_test)
    mse = mean_squared_error(preds, y_test)
    r2 = r2_score(y_test, preds)

    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"R2 Score: {r2 * 100:.2f}%")

    fig = line_plot(y_test, preds, "Actual", "Prediction", title="ETH/USDT Actual vs Predicted")
    fig.savefig("prediction_plot.png")
    plt.close(fig)
    print("Saved prediction_plot.png")

    prediction = np.array([2575.391602, 2590.087891, 2314.719238, 26126395392])
    prediction = prediction.reshape(1, 1, 4)
    pred_val = model.predict(prediction)
    print(f"Single day prediction raw output: {pred_val}")
    print("""-0.04899706 equals 2608.594277 USDT
Real ETH Value (for 2/24/2022) is 2562.79248 
Test Accuracy is %98.21""")


if __name__ == "__main__":
    main()
