# MLFlow-FastAPI

El objetivo de este challenge es entrenar un modelo y utilizar las capacidades de MLFlow para loggear las características de las ejecuciones de todos los experimentos que ejecute, así como el modelo entrenado, y utilizar los resultados para decidir cómo ir variando el valor de los parámetros de entrenamiento del algoritmo para ir mejorando las métricas de funcionamiento y que a la vez comparar el funcionamiento hasta decidir cuál es el que mejor funcionamiento ofrece.

El proceso de este challenge se dividirá en los siguientes pasos:
- [Paso 0](#paso-0): Preparación del entorno:
  - Instalación de dependencias, creación de la carpeta del proyecto, inicialización de Git y DVC y arrancar MLFlow.
- [Paso 1](#paso-1-y-2): Datos:
  - Descargar el dataset y versionarlo con DVC.
- [Paso 2](#paso-1-y-2): Entrenamiento con MLFlow:
  - Creación de script con logging manual de parámetros y métricas. 
- [Paso 3](#paso-3): Experimentos:
  - Lanzar 5 experimentos variando el parámetro necesario. Comparar en MLFlow y decidir a partir de los resultados.
- [Cierre](#cierre): Puesta en producción:
  - Crear el script de la API que carga el modelo y expone el endpoint de predicción. Arrancar con uvicorn.
 
## Dataset
Usaremos el dataset de California Housing incluido en scikit-learn. Tiene 20,640 filas y 8 variables numericas (ingreso medio, edad de la vivienda, habitaciones, etc.) para predecir el precio medio de las casas. No necesita ninguna descarga externa. 
En este Jupyter Notebook está el EDA que realicé para revisar su estructura y confirmar factores extras como distribución o correlaciones importantes. 

## Paso 0
Esta práctica asume que ya se tiene instalado WSL2 con Ubuntu y Miniconda. Si no se han completado esos pasos, en [este repositorio](github.com/adribarra/MLOpsMioti) están las instrucciones 

Una vez completado los pasos anteriores, inicializamos nuestro ambiente virtual:
```
conda activate production_mlops
```
Clonamos el repositorio y creamos una estructura mínima
```
git clone https://github.com/adribarra/MLFlow-FastAPI.git
cd MLFlow-FastAPI
mkdir -p data src notebooks
```

Creamos el archivo de gitignore y agregamos las siguientes lineas para que no se añadan todos los archivos al repositorio:
```
# Entornos virtuales
venv/
.mlops/

# Cachés de Python
__pycache__/
*.py[cod]

# Datos y Modelos (serán gestionados por DVC)
data/
models/

# MLflow local
mlruns/
mlartifacts/

# Secretos
.env
/model.pkl
```

Inicializar DVC:
```
dvc init
git add .dvc .gitignore
git commit -m "init dvc"
```

En una terminal dedicada, arranca el servidor MLflow con soporte para Model Registry:
```
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5000
```
Abrir http://localhost:5000 en el navegador.

## Paso 1 y 2
Antes de crear los archivos para cargar y entrenar el modelo, hice un [Jupyter notebook](https://github.com/adribarra/MLFlow-FastAPI/blob/main/notebooks/CaliforniaHousing_EDA.ipynb) del dataset para entender un poco los datos y su distribución. 
Después del análisis del dataset, pasamos a cargar los datos y versionar con DVC. 

(Ver carpeta de [src](https://github.com/adribarra/MLFlow-FastAPI/tree/main/src) para ver los pasos a seguir)

## Paso 3
Lanzamos tres experimentos variando n_estimators. Cada ejecución queda registrada en MLflow con sus parametros y métricas.
```
# Experimento 1:
python src/train.py --n_estimators 50 --max_depth 10 --run_name 'RF_50_arboles'
# Experimento 2:
python src/train.py --n_estimators 100 --max_depth 10 --run_name 'RF_100_arboles'
# Experimento 3:
python src/train.py --n_estimators 200 --max_depth 10 --run_name 'RF_200_arboles'
```
Al terminar los tres, ir a la interfaz de MLflow, seleccionar el experimento California_Housing y seguir estos pasos:
1. Si las métricas no aparecen, hacer click en Columns > Metrics
2. Hacer click en la columna 'rmse' para ordenar de menor a mayor.
3. Selecciona los tres runs con las casillas de la izquierda.
4. Haz clic en el boton 'Compare' para ver la grafica comparativa.
<img width="1141" height="567" alt="image" src="https://github.com/user-attachments/assets/e9e05a0f-7219-4d37-977a-f56e730dc5c6" />

La línea azul (n_estimators=200) va hacia abajo a la derecha → RMSE más bajo (0.54327), es el mejor modelo.
La línea roja y amarilla (n_estimators=50 y 100) van hacia arriba → RMSE más alto (0.54450-0.54515).
Los tres experimentos con n_estimators=50, 100 y 200 mostraron diferencias de RMSE inferiores a 130 dólares, lo que indica que el modelo converge con pocos árboles en este dataset. Para ver diferencias más significativas podría variar max_depth, para ver si hay un impacto mayor en el RMSE.

```
python src/train.py --n_estimators 100 --max_depth  3 --run_name 'RF_depth3'
python src/train.py --n_estimators 100 --max_depth  5 --run_name 'RF_depth5'
python src/train.py --n_estimators 100 --max_depth 20 --run_name 'RF_depth20'
```
<img width="1134" height="567" alt="image" src="https://github.com/user-attachments/assets/2c93d300-02db-4984-abb9-948fecc909bd" />

Con esto podemos identificar que max_depth es el parámetro crítico. Aumentar max_depth de 3 a 20 redujo el RMSE en más de 30.000 dólares, mientras que variar n_estimators apenas tuvo impacto. 
En este caso, el modelo óptimo fue n_estimators=100, max_depth=20.

## Cierre
Volver a la carpeta de [src](https://github.com/adribarra/MLFlow-FastAPI/tree/main/src) e ir al paso de la creación de src/api.py en caso de no haber sido creado todavía.
Arrancar la API:
```
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Deberas ver:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

Abrir una tercera terminal y ejecutar estos comandos:

Para verificar que la API responde:
```
curl http://localhost:8000/
```
Respuesta esperada:
```
{"status":"ok","modelo":"RandomForest California Housing"}
```
Predicción 1: barrio de ingreso alto y zona costera 
```
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "MedInc": 8.5,
    "HouseAge": 15.0,
    "AveRooms": 7.0,
    "AveBedrms": 1.2,
    "Population": 800.0,
    "AveOccup": 2.5,
    "Latitude": 37.8,
    "Longitude": -122.4
  }'
```
Respuesta:
```
{"predicted_value":4.458,"predicted_price_usd":"$445,795"}
```

Predicción 2: barrio de ingreso bajo e interior
```
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "MedInc": 1.5,
    "HouseAge": 40.0,
    "AveRooms": 4.0,
    "AveBedrms": 1.1,
    "Population": 2000.0,
    "AveOccup": 4.0,
    "Latitude": 36.7,
    "Longitude": -119.8
  }'
```
Respuesta:
```
{"predicted_value":0.5152,"predicted_price_usd":"$51,524"}
```

Predicción 3: valores por defecto definidos en el script
```
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{}'
```
Respuesta:
```
{"predicted_value":0.854,"predicted_price_usd":"$85,401"}
```
