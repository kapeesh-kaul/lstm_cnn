import torch
import torch.nn as nn
import numpy as np


class LSTMModel(nn.Module):
    """
    LSTM Model for Stock Price Prediction.

    Parameters:
    - input_size: Number of input features (1 for univariate time series).
    - hidden_size: Number of hidden units in the LSTM.
    - num_layers: Number of LSTM layers.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass for the model.

        Parameters:
        - x: Input tensor of shape [batch_size, sequence_length, input_size].

        Returns:
        - Output tensor of shape [batch_size, 1].
        """
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])  # Use the output at the last time step
        return output


def train_model(model, criterion, optimizer, train_loader, num_epochs, device):
    """
    Train the LSTM model.

    Parameters:
    - model: The LSTM model instance.
    - criterion: Loss function (e.g., MSELoss).
    - optimizer: Optimizer (e.g., Adam).
    - train_loader: DataLoader for training data.
    - num_epochs: Number of training epochs.
    - device: Device to use ('cuda' or 'cpu').
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            # Move data to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))  # Ensure y_batch has shape [batch_size, 1]

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def evaluate_model(model, test_loader, scaler, device):
    """
    Evaluate the model and return predictions and actual values.

    Parameters:
    - model: The LSTM model instance.
    - test_loader: DataLoader for testing data.
    - scaler: Fitted scaler to inverse transform predictions and actuals.
    - device: Device to use ('cuda' or 'cpu').

    Returns:
    - predictions: Predicted stock prices (numpy array).
    - actuals: Actual stock prices (numpy array).
    """
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # Move data to device
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(x_batch).cpu().numpy()
            predictions.extend(outputs)

            # Collect actual values
            actuals.extend(y_batch.view(-1, 1).cpu().numpy())  # Reshape y_batch to [batch_size, 1]
    
    # Inverse transform predictions and actuals
    predictions = scaler.inverse_transform(np.array(predictions))
    actuals = scaler.inverse_transform(np.array(actuals))
    return predictions, actuals
