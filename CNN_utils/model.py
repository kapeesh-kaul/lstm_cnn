import torch
import torch.nn as nn
import numpy as np

class CNNModel(nn.Module):
    """
    CNN Model for Stock Price Prediction.
    """
    def __init__(self, input_channels=1, num_filters=64, kernel_size=3, fc_units=128):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * (60 - kernel_size + 1), fc_units)  # Adjust dimensions for kernel size
        self.output = nn.Linear(fc_units, 1)

    def forward(self, x):
        """
        Forward pass for the CNN.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.output(x)
        return x


def train_model(model, criterion, optimizer, train_loader, num_epochs, device):
    """Train the CNN model."""
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))  # Reshape target to match output shape
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def evaluate_model(model, test_loader, scaler, device):
    """Evaluate the CNN model."""
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).cpu().numpy()
            predictions.extend(outputs)
            actuals.extend(y_batch.view(-1, 1).cpu().numpy())
    
    predictions = scaler.inverse_transform(np.array(predictions))
    actuals = scaler.inverse_transform(np.array(actuals))
    return predictions, actuals
