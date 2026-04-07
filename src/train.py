# src/train.py
import argparse
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Apuntar al servidor MLflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('California_Housing')

def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument('--n_estimators', type=int, default=100)
	p.add_argument('--max_depth', type=int, default=10)
	p.add_argument('--run_name', type=str, default=None)
	return p.parse_args()

def run_training(args):
	run_name = args.run_name or f'RF_n{args.n_estimators}_d{args.max_depth}'
	# Cargar datos
	df = pd.read_csv('data/housing.csv')
	X = df.drop(columns=['MedHouseVal'])
	y = df['MedHouseVal']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	with mlflow.start_run(run_name=run_name):
	
		# 1. LOGGEAR PARAMETROS
		mlflow.log_param('n_estimators', args.n_estimators)
		mlflow.log_param('max_depth', args.max_depth)
		mlflow.log_param('train_size', len(X_train))
		mlflow.log_param('test_size', len(X_test))

		# 2. ENTRENAR
		model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=-1, random_state=42)
		model.fit(X_train, y_train)

		# 3. CALCULAR METRICAS
		y_pred = model.predict(X_test)
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))

		# Validacion cruzada sobre el set de entrenamiento
		cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error')
		cv_rmse = -cv_scores.mean()

		# 4. LOGGEAR METRICAS
		mlflow.log_metric('rmse', rmse)
		mlflow.log_metric('cv_rmse', cv_rmse)
		print(f'[{run_name}] RMSE={rmse:.4f} CV_RMSE={cv_rmse:.4f}')

		# 5. LOGGEAR EL MODELO
		mlflow.sklearn.log_model(model, artifact_path='model')

		# Guardar tambien como .pkl local para la API
		with open('model.pkl', 'wb') as f:
			pickle.dump(model, f)
		return rmse

if __name__ == '__main__':
	args = parse_args()
	run_training(args)
