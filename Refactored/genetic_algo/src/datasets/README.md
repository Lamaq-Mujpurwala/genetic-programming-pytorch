# Dataset Management System

This module provides a flexible and extensible system for managing datasets in the genetic algorithm project. It includes support for various datasets and provides consistent preprocessing capabilities.

## Structure

```
datasets/
├── __init__.py
├── base_dataset.py      # Base class for all dataset handlers
├── preprocessing.py     # Data preprocessing utilities
├── dataset_factory.py   # Central registry for all datasets
└── wine_dataset.py      # Wine dataset implementation
```

## Usage Examples

### Basic Usage

```python
from genetic_algo.src.datasets import DatasetFactory, DataPreprocessor

# Get the dataset factory
factory = DatasetFactory()

# Get the Wine dataset
wine_dataset = factory.get_dataset("wine")

# Get the data
data = wine_dataset.get_data()

# Preprocess the data
preprocessor = DataPreprocessor()
X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.prepare_data(
    data, 
    target_column=wine_dataset.get_target_name()
)
```

### Getting Dataset Information

```python
# Get information about the dataset
dataset_info = wine_dataset.get_dataset_info()
print(f"Dataset: {dataset_info['name']}")
print(f"Number of samples: {dataset_info['n_samples']}")
print(f"Number of features: {dataset_info['n_features']}")
print(f"Feature names: {dataset_info['feature_names']}")
```

### Adding a New Dataset

To add a new dataset (e.g., Bean dataset), create a new file `bean_dataset.py`:

```python
from .base_dataset import BaseDataset
import pandas as pd

class BeanDataset(BaseDataset):
    def __init__(self):
        super().__init__(name="Bean", target_column="target")
    
    def load_data(self) -> pd.DataFrame:
        # Implement your data loading logic here
        # Return a pandas DataFrame with features and target
        pass
```

Then register it in the `DatasetFactory`:

```python
from .bean_dataset import BeanDataset

# In DatasetFactory.__init__
self._datasets["bean"] = BeanDataset
```

## Available Datasets

Currently supported datasets:
- Wine dataset (from scikit-learn)

## Preprocessing Features

The `DataPreprocessor` class provides:
- Train/test splitting
- Feature scaling using StandardScaler
- Consistent preprocessing across all datasets

## Adding New Datasets

To add a new dataset:
1. Create a new class inheriting from `BaseDataset`
2. Implement the required methods
3. Register the dataset in `DatasetFactory`
4. Update this documentation

## Best Practices

1. Always use the `DatasetFactory` to access datasets
2. Use the `DataPreprocessor` for consistent preprocessing
3. Document any dataset-specific requirements or preprocessing steps
4. Include dataset information and statistics in the dataset class 