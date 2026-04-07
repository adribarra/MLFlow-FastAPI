# src/load_data.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def download_data():
    os.makedirs('data', exist_ok=True)
    dataset = fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['MedHouseVal'] = dataset.target
    df.to_csv('data/housing.csv', index=False)
    print(f'Dataset guardado: {df.shape[0]} filas, {df.shape[1]} columnas')

if __name__ == '__main__':
    download_data()
