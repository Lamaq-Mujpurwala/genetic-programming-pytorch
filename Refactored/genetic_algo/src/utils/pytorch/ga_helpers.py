import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .models import get_model_architecture

def get_weights_pytorch(dataset_name: str, input_size: int = None, num_classes: int = None):
    """Get initial weights for the model.
    
    Args:
        dataset_name (str): Name of the dataset
        input_size (int): Size of input features (for MLP)
        num_classes (int): Number of output classes
        
    Returns:
        numpy.ndarray: Flattened weights
    """
    model = get_model_architecture(dataset_name, input_size, num_classes)
    weights_list = []
    for param in model.parameters():
        weights_list.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(weights_list)

def reassemble_model_pytorch(individual, dataset_name: str, input_size: int = None, 
                           num_classes: int = None, device: str = None):
    """Reassemble model from individual weights.
    
    Args:
        individual: DEAP individual containing weights
        dataset_name (str): Name of the dataset
        input_size (int): Size of input features (for MLP)
        num_classes (int): Number of output classes
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        nn.Module: Reassembled model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract flat vector from DEAP individual
    flat_vector = individual[0]
    
    # Create model instance
    model = get_model_architecture(dataset_name, input_size, num_classes)
    
    # Set weights
    model.set_weights(torch.tensor(flat_vector, dtype=torch.float32))
    
    # Move model to device
    model = model.to(device)
    
    return model

def get_fitness_pytorch(individual, X_train_scaled, X_test_scaled, y_train, y_test,
                       dataset_name: str, input_size: int = None, num_classes: int = None,
                       device: str = None):
    """Calculate fitness (loss) for an individual.
    
    Args:
        individual: DEAP individual containing weights
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled testing features
        y_train: Training labels
        y_test: Testing labels
        dataset_name (str): Name of the dataset
        input_size (int): Size of input features (for MLP)
        num_classes (int): Number of output classes
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (loss,) for DEAP compatibility
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = reassemble_model_pytorch(individual, dataset_name, input_size, num_classes, device)
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Training
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(20):  # Fixed number of epochs
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            num_batches += 1
    
    avg_test_loss = total_loss / num_batches
    return (avg_test_loss,)

def mutation_pytorch(offspring, X_train_scaled, y_train, dataset_name: str,
                    input_size: int = None, num_classes: int = None, device: str = None):
    """Apply mutation to an individual through brief training.
    
    Args:
        offspring: Individual to mutate
        X_train_scaled: Scaled training features
        y_train: Training labels
        dataset_name (str): Name of the dataset
        input_size (int): Size of input features (for MLP)
        num_classes (int): Number of output classes
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        Individual: Mutated individual
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Handle Individual objects
    if hasattr(offspring, 'fitness'):
        individual_to_mutate = offspring
    else:
        from deap import creator
        individual_to_mutate = creator.Individual([offspring])
    
    model = reassemble_model_pytorch(individual_to_mutate, dataset_name, input_size, num_classes, device)
    
    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Brief training for mutation
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):  # Short training for mutation
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
    
    # Extract weights after training
    weights_list = []
    for param in model.parameters():
        weights_list.append(param.detach().cpu().numpy().flatten())
    
    flattened_weights = np.concatenate(weights_list)
    
    # Create new Individual with mutated weights
    from deap import creator
    mutated_individual = creator.Individual([flattened_weights])
    
    return mutated_individual

def check_device():
    """Check and return the appropriate device for PyTorch.
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        device = 'cpu'
        print("GPU not available, using CPU")
    
    return device 