from sklearn.datasets import load_wine
import pandas as pd

#This is just for abstraction , the actual get_dataset.get_scaled_data directly takes a pandas dataFrame

def get_wine_data():
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target
    wine = pd.DataFrame(data=X, columns=wine_data.feature_names)
    wine['target'] = y

    return wine