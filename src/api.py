# src/api.py
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Cargar el modelo al arrancar (una sola vez)
with open('model.pkl', 'rb') as f:
	model = pickle.load(f)
app = FastAPI(title='California Housing Price API')

# Schema de entrada: las 8 variables del dataset
class HouseFeatures(BaseModel):
	MedInc: float = 3.9 # Ingreso medio del barrio
	HouseAge: float = 28.6 # Edad media de la vivienda
	AveRooms: float = 5.4 # Habitaciones promedio
	AveBedrms: float = 1.1 # Dormitorios promedio
	Population: float = 1425.5 # Poblacion del bloque
	AveOccup: float = 3.1 # Ocupacion media
	Latitude: float = 35.6 # Latitud
	Longitude: float = -119.6 # Longitud

@app.get('/')
def root():
	return {'status': 'ok', 'modelo': 'RandomForest California Housing'}

@app.post('/predict')
def predict(features: HouseFeatures):
	X = [[
	features.MedInc, features.HouseAge, features.AveRooms,
	features.AveBedrms, features.Population, features.AveOccup,
	features.Latitude, features.Longitude
	]]
	prediction = model.predict(X)[0]
	return {
		'predicted_value': round(float(prediction), 4),
		'predicted_price_usd': f'${prediction * 100_000:,.0f}'
	}
