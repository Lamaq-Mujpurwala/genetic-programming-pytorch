import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# This is also for Abstraction since i dont want to write the entrie loading and scaling part in the main code

def prepare_data_for_scaling(dataset, target_column):
  """
  Prepares data for scaling by splitting into features and target,
  and then into training and testing sets.

  Args:
    dataset: pandas DataFrame, the input dataset.
    target_column: str, the name of the target column.

  Returns:
    tuple: (X_train, X_test, y_train, y_test)
  """
  X = dataset.drop(columns=[target_column])
  y = dataset[target_column]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

def get_scaled_data(dataset, target_column):
  """
  Scales the features of a dataset and returns the scaled training
  and testing sets along with the original target sets.

  Args:
    dataset: pandas DataFrame, the input dataset.
    target_column: str, the name of the target column.

  Returns:
    tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
  """
  X_train, X_test, y_train, y_test = prepare_data_for_scaling(dataset, target_column)

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
  # Example usage (this part will only run when the script is executed directly)
  from sklearn.datasets import load_wine

  # Load a sample dataset (Wine dataset)
  wine = load_wine()
  wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
  wine_df['target'] = wine.target

  # Specify the target column
  target_column_name = 'target'

  # Get the scaled data
  X_train_scaled, X_test_scaled, y_train, y_test = get_scaled_data(wine_df, target_column_name)

  # Print shapes to verify
  print("Shape of X_train_scaled:", X_train_scaled.shape)
  print("Shape of X_test_scaled:", X_test_scaled.shape)
  print("Shape of y_train:", y_train.shape)
  print("Shape of y_test:", y_test.shape)

  # You can now use X_train_scaled, X_test_scaled, y_train, y_test for model training