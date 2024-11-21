import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from CNN_utils import load_data, preprocess_data, create_sequences
from CNN_utils import CNNModel, evaluate_model
import numpy as np

# Configuration
PREDICT_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ticker": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "seq_length": 60,
    "batch_size": 64,
    "num_filters": 64,
    "kernel_size": 3,
    "fc_units": 128,
    "model_load_path": "cnn_model.pth",
    "results_save_path": "CNN_predictions.csv"
}


def prepare_test_data(config):
    """Load, preprocess, and prepare data for testing."""
    data = load_data(config["ticker"], config["start_date"], config["end_date"])
    data, scaler = preprocess_data(data)
    dataset = data['Close'].values
    x, y = create_sequences(dataset, config["seq_length"])
    train_size = int(len(x) * 0.8)
    x_test, y_test = x[train_size:], y[train_size:]
    x_test = torch.tensor(x_test.transpose((0, 2, 1)), dtype=torch.float32)  # Corrected shape
    y_test = torch.tensor(y_test, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=config["batch_size"], shuffle=False)
    return test_loader, scaler


def predict(config):
    """Predict stock prices using the trained CNN model."""
    test_loader, scaler = prepare_test_data(config)

    model = CNNModel(input_channels=1, num_filters=config["num_filters"], kernel_size=config["kernel_size"], fc_units=config["fc_units"])
    model.load_state_dict(torch.load(config["model_load_path"], map_location=config["device"], weights_only=True))
    model.to(config["device"])

    predictions, actuals = evaluate_model(model, test_loader, scaler, config["device"])

    results_df = pd.DataFrame({"Actual": actuals.flatten(), "Predicted": predictions.flatten()})
    results_df.to_csv(config["results_save_path"], index=False)
    print(f"Predictions saved to {config['results_save_path']}")


if __name__ == "__main__":
    predict(PREDICT_CONFIG)
