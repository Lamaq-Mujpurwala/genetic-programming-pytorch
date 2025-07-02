from typing import Dict, Type
from .base_dataset import BaseDataset
from .wine_dataset import WineDataset

class DatasetFactory:
    """Factory class for managing and accessing different datasets.
    
    This class provides a centralized way to access different datasets
    and ensures consistent dataset handling across the application.
    
    Attributes:
        _datasets (Dict[str, Type[BaseDataset]]): Dictionary mapping dataset names to their handler classes
    """
    
    def __init__(self):
        """Initialize the dataset factory with available datasets."""
        self._datasets: Dict[str, Type[BaseDataset]] = {
            "wine": WineDataset,
            # Add more datasets here as they are implemented
            # "bean": BeanDataset,
            # "higgs": HiggsDataset,
            # "cifar": CIFARDataset,
        }
    
    def get_dataset(self, dataset_name: str) -> BaseDataset:
        """Get a dataset handler instance by name.
        
        Args:
            dataset_name (str): Name of the dataset to get
            
        Returns:
            BaseDataset: An instance of the requested dataset handler
            
        Raises:
            ValueError: If the requested dataset is not available
        """
        dataset_name = dataset_name.lower()
        if dataset_name not in self._datasets:
            available = ", ".join(self._datasets.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {available}")
        
        return self._datasets[dataset_name]()
    
    def list_available_datasets(self) -> list:
        """Get a list of all available dataset names.
        
        Returns:
            list: List of available dataset names
        """
        return list(self._datasets.keys())
    
    def register_dataset(self, name: str, dataset_class: Type[BaseDataset]):
        """Register a new dataset handler.
        
        Args:
            name (str): Name to register the dataset under
            dataset_class (Type[BaseDataset]): The dataset handler class
        """
        if not issubclass(dataset_class, BaseDataset):
            raise TypeError("Dataset class must inherit from BaseDataset")
        self._datasets[name.lower()] = dataset_class 