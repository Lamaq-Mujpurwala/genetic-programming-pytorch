import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# --- Model Weight Helpers ---
def get_model_weights(model) -> np.ndarray:
    """
    Flatten and return the weights of a given model as a single vector.
    Args:
        model: The model instance (nn.Module) whose weights are to be flattened.
    Returns:
        A 1D numpy array of all model parameters concatenated.
    """
    flattened_weights = []
    for param in model.parameters():
        flattened_weights.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(flattened_weights)


def reassemble_model(flat_vector, model_class, device, *model_args, **model_kwargs):
    """
    Reconstruct a PyTorch model from a flattened weight vector.
    Args:
        flat_vector: Flattened weight vector (numpy array).
        model_class: The nn.Module class to instantiate.
        device: PyTorch device.
        *model_args, **model_kwargs: Arguments for model instantiation.
    Returns:
        The model instance with weights set from the flat vector.
    """
    model = model_class(*model_args, **model_kwargs).to(device)
    param_shapes = [param.shape for param in model.parameters()]
    param_sizes = [np.prod(shape) for shape in param_shapes]
    params = []
    idx = 0
    for shape, size in zip(param_shapes, param_sizes):
        param_data = torch.FloatTensor(flat_vector[idx:idx+size].reshape(shape)).to(device)
        params.append(param_data)
        idx += size
    with torch.no_grad():
        for param, new_data in zip(model.parameters(), params):
            param.data.copy_(new_data)
    return model

# --- Fitness Evaluation ---
def get_ga_fitness(individual, reassemble_model_func, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device, epochs=20, patience=20):
    """
    Evaluate the fitness of an individual (flattened weight vector) by training and testing a model.
    Args:
        individual: Flattened weight vector.
        reassemble_model_func: Function to reconstruct the model from the weight vector.
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor: Data tensors.
        device: PyTorch device.
        epochs: Number of training epochs.
        patience: Early stopping patience.
    Returns:
        Test loss as a float (lower is better).
    """
    try:
        model = reassemble_model_func(individual)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        best_loss = float('inf')
        patience_counter = 0
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            avg_loss = total_loss / num_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
        return test_loss.item()
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return float('inf')


def get_fitness(individual, *args, **kwargs):
    """
    Alias for get_ga_fitness for compatibility with other codebases.
    Passes all arguments to get_ga_fitness.
    """
    return get_ga_fitness(individual, *args, **kwargs) 