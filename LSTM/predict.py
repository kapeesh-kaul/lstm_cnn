import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from LSTM_utils import load_data, preprocess_data, create_sequences
from LSTM_utils import LSTMModel, evaluate_model

# Configuration
PREDICT_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ticker": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "seq_length": 60,
    "batch_size": 64,
    "hidden_size": 64,
    "num_layers": 2,
    "model_load_path": "lstm_model.pth",
    "results_save_path": "LSTM_predictions.csv"
}


def prepare_test_data(config):
    """Load, preprocess, and create sequences for testing."""
    data = load_data(config["ticker"], config["start_date"], config["end_date"])
    data, scaler = preprocess_data(data)
    dataset = data['Close'].values
    x, y = create_sequences(dataset, config["seq_length"])
    train_size = int(len(x) * 0.8)
    x_test, y_test = x[train_size:], y[train_size:]
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=config["batch_size"], shuffle=False)
    return test_loader, scaler


def predict(config):
    """Load the trained model, run predictions, and save results to a CSV."""
    # Prepare test data
    test_loader, scaler = prepare_test_data(config)

    # Load the trained model
    model = LSTMModel(input_size=1, hidden_size=config["hidden_size"], num_layers=config["num_layers"])
    model.load_state_dict(torch.load(config["model_load_path"], map_location=config["device"], weights_only=True))
    model.to(config["device"])
    print(f"Model weights loaded from {config['model_load_path']}")

    # Evaluate the model
    predictions, actuals = evaluate_model(model, test_loader, scaler, config["device"])

    # Save predictions to a CSV file
    results_df = pd.DataFrame({
        "Actual": actuals.flatten(),
        "Predicted": predictions.flatten()
    })
    results_df.to_csv(config["results_save_path"], index=False)
    print(f"Predictions saved to {config['results_save_path']}")


if __name__ == "__main__":
    predict(PREDICT_CONFIG)
