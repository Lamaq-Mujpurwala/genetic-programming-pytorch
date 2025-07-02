from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Handles data preprocessing operations like scaling and splitting.
    
    This class provides methods for common preprocessing tasks such as:
    - Splitting data into training and testing sets
    - Scaling features
    - Handling missing values
    
    Attributes:
        scaler (StandardScaler): The scaler instance used for feature scaling
    """
    
    def __init__(self):
        """Initialize the preprocessor with a StandardScaler."""
        self.scaler = StandardScaler()
    
    def prepare_data(self, 
                    dataset: pd.DataFrame, 
                    target_column: str,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data by splitting and scaling.
        
        Args:
            dataset (pd.DataFrame): The input dataset
            target_column (str): Name of the target column
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple containing:
                - X_train_scaled: Scaled training features
                - X_test_scaled: Scaled testing features
                - y_train: Training target values
                - y_test: Testing target values
        """
        # Split features and target
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_scaler(self) -> StandardScaler:
        """Get the fitted scaler instance.
        
        Returns:
            StandardScaler: The fitted scaler
        """
        return self.scaler 