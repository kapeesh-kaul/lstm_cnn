import torch
from torch.utils.data import DataLoader, TensorDataset
from LSTM_utils import load_data, preprocess_data, create_sequences
from LSTM_utils import LSTMModel, train_model
import os

# Configuration
TRAIN_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "ticker": "AAPL",
    "start_date": "2015-01-01",
    "end_date": "2023-12-31",
    "seq_length": 60,
    "batch_size": 64,
    "hidden_size": 64,
    "num_layers": 2,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "model_save_path": "lstm_model.pth"
}


def prepare_data(config):
    """Load, preprocess, and create sequences for training."""
    data = load_data(config["ticker"], config["start_date"], config["end_date"])
    data, scaler = preprocess_data(data)
    dataset = data['Close'].values
    x, y = create_sequences(dataset, config["seq_length"])
    train_size = int(len(x) * 0.8)
    x_train, y_train = x[:train_size], y[:train_size]
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config["batch_size"], shuffle=True)
    return train_loader, scaler


def train(config):
    """Train the LSTM model and save the weights."""
    # Prepare data
    train_loader, _ = prepare_data(config)

    # Initialize model, criterion, and optimizer
    model = LSTMModel(input_size=1, hidden_size=config["hidden_size"], num_layers=config["num_layers"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train the model
    train_model(model, criterion, optimizer, train_loader, config["num_epochs"], config["device"])

    # Save the trained model weights
    save_dir = os.path.dirname(config["model_save_path"])
    if save_dir:  # Only create directories if the path includes them
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model weights saved to {config['model_save_path']}")


if __name__ == "__main__":
    train(TRAIN_CONFIG)
