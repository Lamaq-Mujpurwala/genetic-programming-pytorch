import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(nn.Module):
    """Base class for all neural network models.
    
    This class defines the interface that all model architectures must implement.
    It provides common functionality and ensures consistent behavior across different architectures.
    
    Attributes:
        input_size (int): Size of input features
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, input_size: int, num_classes: int, dropout_rate: float = 0.3):
        """Initialize the base model.
        
        Args:
            input_size (int): Size of input features
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
    
    def get_weights(self) -> torch.Tensor:
        """Get all model weights as a flattened tensor.
        
        Returns:
            torch.Tensor: Flattened weights
        """
        weights_list = []
        for param in self.parameters():
            weights_list.append(param.detach().cpu().numpy().flatten())
        return torch.tensor(np.concatenate(weights_list))
    
    def set_weights(self, flat_weights: torch.Tensor):
        """Set model weights from a flattened tensor.
        
        Args:
            flat_weights (torch.Tensor): Flattened weights to set
        """
        start_idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_shape = param.shape
            param_weights = flat_weights[start_idx:start_idx + param_size].reshape(param_shape)
            param.data = param_weights
            start_idx += param_size

class MLPModel(BaseModel):
    """Multi-layer Perceptron model for tabular data (Wine, Bean, Higgs).
    
    This model uses a standard MLP architecture with multiple fully connected layers
    and dropout for regularization. Suitable for tabular datasets.
    
    Attributes:
        hidden_sizes (list): List of hidden layer sizes
    """
    
    def __init__(self, input_size: int, num_classes: int, 
                 hidden_sizes: list = [128, 64, 32], dropout_rate: float = 0.3):
        """Initialize the MLP model.
        
        Args:
            input_size (int): Size of input features
            num_classes (int): Number of output classes
            hidden_sizes (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__(input_size, num_classes, dropout_rate)
        self.hidden_sizes = hidden_sizes
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output probabilities
        """
        x = self.layers(x)
        return F.softmax(x, dim=1)

class CNNModel(BaseModel):
    """Convolutional Neural Network model for image data (CIFAR, Thumbnail).
    
    This model uses a CNN architecture suitable for image classification tasks.
    It includes convolutional layers, pooling, and fully connected layers.
    
    Attributes:
        in_channels (int): Number of input channels (e.g., 3 for RGB)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 10, dropout_rate: float = 0.3):
        """Initialize the CNN model.
        
        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__(in_channels, num_classes, dropout_rate)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size of flattened features
        self.flat_features = 128 * 4 * 4  # Adjust based on input size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output probabilities
        """
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, self.flat_features)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

def get_model_architecture(dataset_name: str, input_size: int = None, num_classes: int = None):
    """Factory function to get the appropriate model architecture.
    
    Args:
        dataset_name (str): Name of the dataset
        input_size (int): Size of input features (for MLP)
        num_classes (int): Number of output classes
        
    Returns:
        BaseModel: Appropriate model instance
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name.lower() in ['wine', 'bean', 'higgs']:
        if input_size is None or num_classes is None:
            raise ValueError("input_size and num_classes are required for MLP models")
        return MLPModel(input_size=input_size, num_classes=num_classes)
    
    elif dataset_name.lower() in ['cifar', 'thumbnail']:
        if num_classes is None:
            raise ValueError("num_classes is required for CNN models")
        return CNNModel(num_classes=num_classes)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}") 