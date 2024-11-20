import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNgramModel(nn.Module):
    def __init__(self, num_classes=36, hidden_dim=128, n_grams=2):
        super(CNNNgramModel, self).__init__()
        
        # Convolutional layers (similar to your original model)
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # LSTM layer (for n-gram sequence processing)
        self.lstm = nn.LSTM(input_size=64 * 7, hidden_size=hidden_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        
        # Flatten for LSTM input, treating height as sequence length
        x = x.permute(0, 2, 1, 3).flatten(2)  # shape: (batch, height, width * channels)
        
        # LSTM layer processes flattened image patches as sequences
        lstm_out, _ = self.lstm(x)  # shape: (batch, height, hidden_dim)
        
        # Get the last output in the LSTM sequence
        x = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
