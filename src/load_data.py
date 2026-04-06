# src/load_data.py — alternativa sin fetch
import pandas as pd
import os

def download_data():
    os.makedirs('data', exist_ok=True)
    
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['ocean_proximity'])
    df.to_csv('data/housing.csv', index=False)
    print(f'Dataset guardado: {df.shape[0]} filas, {df.shape[1]} columnas')

if __name__ == '__main__':
    download_data()
