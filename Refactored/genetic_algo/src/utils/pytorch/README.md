# PyTorch Utilities for Genetic Algorithm

This module provides PyTorch-based utilities for implementing genetic algorithms with neural networks. It includes model architectures and helper functions for the GA operations.

## Structure

```
pytorch/
├── models.py          # Neural network model architectures
├── ga_helpers.py      # GA helper functions
└── README.md         # This file
```

## Model Architectures

### BaseModel
The base class for all neural network models. Provides common functionality for:
- Weight management (getting/setting weights)
- Basic model interface
- Common attributes (input size, number of classes, dropout rate)

### MLPModel
Multi-layer Perceptron for tabular data (Wine, Bean, Higgs datasets).
- Multiple fully connected layers
- ReLU activation
- Dropout for regularization
- Configurable hidden layer sizes

### CNNModel
Convolutional Neural Network for image data (CIFAR, Thumbnail datasets).
- Convolutional layers with ReLU activation
- Max pooling
- Dropout for regularization
- Fully connected layers for classification

## GA Helper Functions

### Model Management
- `get_weights_pytorch`: Get initial weights for a model
- `reassemble_model_pytorch`: Reconstruct model from individual weights

### Fitness and Mutation
- `get_fitness_pytorch`: Calculate fitness (loss) for an individual
- `mutation_pytorch`: Apply mutation through brief training

### Utility Functions
- `check_device`: Check and return appropriate device (CPU/GPU)

## Usage Examples

### Basic Usage
```python
from utils.pytorch.ga_helpers import (
    get_weights_pytorch,
    reassemble_model_pytorch,
    get_fitness_pytorch,
    mutation_pytorch,
    check_device
)

# Initialize weights for Wine dataset
weights = get_weights_pytorch("wine", input_size=13, num_classes=3)

# Get fitness for an individual
fitness = get_fitness_pytorch(
    individual,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    dataset_name="wine",
    input_size=13,
    num_classes=3
)
```

### Using Different Model Architectures
```python
# For tabular data (Wine, Bean, Higgs)
weights = get_weights_pytorch("wine", input_size=13, num_classes=3)
weights = get_weights_pytorch("bean", input_size=16, num_classes=7)
weights = get_weights_pytorch("higgs", input_size=28, num_classes=2)

# For image data (CIFAR, Thumbnail)
weights = get_weights_pytorch("cifar", num_classes=10)
weights = get_weights_pytorch("thumbnail", num_classes=10)
```

## Model Configuration

### MLP Configuration
```python
model = MLPModel(
    input_size=13,
    num_classes=3,
    hidden_sizes=[128, 64, 32],  # Customize hidden layers
    dropout_rate=0.3
)
```

### CNN Configuration
```python
model = CNNModel(
    in_channels=3,  # RGB images
    num_classes=10,
    dropout_rate=0.3
)
```

## Adding New Models

To add a new model architecture:
1. Create a new class inheriting from `BaseModel`
2. Implement required methods (`forward`, `get_weights`, `set_weights`)
3. Add the model to the `get_model_architecture` factory function

Example:
```python
class NewModel(BaseModel):
    def __init__(self, input_size, num_classes, dropout_rate=0.3):
        super().__init__(input_size, num_classes, dropout_rate)
        # Add your model architecture here
    
    def forward(self, x):
        # Implement forward pass
        pass
```

## Best Practices

1. **Model Selection**:
   - Use MLP for tabular data
   - Use CNN for image data
   - Consider dataset characteristics when choosing architecture

2. **Memory Management**:
   - Use `check_device()` to ensure proper GPU usage
   - Clear GPU memory when needed
   - Monitor memory usage during training

3. **Performance**:
   - Use appropriate batch sizes
   - Consider using mixed precision training for large models
   - Profile memory usage for large datasets

4. **Error Handling**:
   - Always check device availability
   - Handle out-of-memory situations
   - Validate input shapes and types 