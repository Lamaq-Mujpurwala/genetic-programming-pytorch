from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Any, Dict
import numpy as np

class BaseDataset(ABC):
    """Base class for all dataset handlers.
    
    This abstract base class defines the interface that all dataset handlers must implement.
    It provides a standardized way to load, preprocess, and access datasets.
    
    Attributes:
        name (str): Name of the dataset
        target_column (str): Name of the target column in the dataset
        feature_columns (list): List of feature column names
        data (pd.DataFrame): The loaded dataset
    """
    
    def __init__(self, name: str, target_column: str):
        """Initialize the dataset handler.
        
        Args:
            name (str): Name of the dataset
            target_column (str): Name of the target column
        """
        self.name = name
        self.target_column = target_column
        self.feature_columns = []
        self.data = None
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from its source.
        
        Returns:
            pd.DataFrame: The loaded dataset
        """
        pass
    
    def get_data(self) -> pd.DataFrame:
        """Get the dataset, loading it if necessary.
        
        Returns:
            pd.DataFrame: The dataset
        """
        if self.data is None:
            self.data = self.load_data()
            self.feature_columns = [col for col in self.data.columns if col != self.target_column]
        return self.data
    
    def get_feature_names(self) -> list:
        """Get the names of all feature columns.
        
        Returns:
            list: List of feature column names
        """
        if not self.feature_columns:
            self.get_data()
        return self.feature_columns
    
    def get_target_name(self) -> str:
        """Get the name of the target column.
        
        Returns:
            str: Name of the target column
        """
        return self.target_column 