import pandas as pd
from sklearn.datasets import load_wine
from .base_dataset import BaseDataset

class WineDataset(BaseDataset):
    """Handler for the Wine dataset from scikit-learn.
    
    This class provides access to the Wine dataset, which contains chemical
    measurements of different wines and their corresponding classes.
    
    Attributes:
        Inherits all attributes from BaseDataset
    """
    
    def __init__(self):
        """Initialize the Wine dataset handler."""
        super().__init__(name="Wine", target_column="target")
    
    def load_data(self) -> pd.DataFrame:
        """Load the Wine dataset from scikit-learn.
        
        Returns:
            pd.DataFrame: The Wine dataset with features and target
        """
        wine_data = load_wine()
        df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
        df[self.target_column] = wine_data.target
        return df
    
    def get_dataset_info(self) -> dict:
        """Get information about the Wine dataset.
        
        Returns:
            dict: Dictionary containing dataset information
        """
        return {
            "name": self.name,
            "n_samples": len(self.get_data()),
            "n_features": len(self.get_feature_names()),
            "feature_names": self.get_feature_names(),
            "target_name": self.target_column,
            "description": "Wine dataset containing chemical measurements of different wines"
        } 