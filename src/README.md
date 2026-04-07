## Script para carga de datos
Usamos el dataset de California Housing incluido en scikit-learn. Tiene 20.640 filas y 8 variables numéricas (ingreso medio, edad de la vivienda, habitaciones, etc.) para predecir el precio medio de las casas.\
En el script [load_data.py](https://github.com/adribarra/MLFlow-FastAPI/blob/main/src/load_data.py) cargamos los datos y pedimos que nos de el número de filas y columnas.
```
python src/load_data.py
```
Output: 
```
Dataset guardado: 20640 filas, 9 columnas
```
Versionamos con DVC:
```
dvc add data/housing.csv
git add data/housing.csv.dvc data/.gitignore
git commit -m 'Dataset California Housing'
```

## Script de entrenamiento
Este es el nucleo de la práctica. El script acepta 'n_estimators' como argumento para poder lanzar varios experimentos con distinto valor y compararlos en MLflow.\
En el script [train.py](https://github.com/adribarra/MLFlow-FastAPI/blob/main/src/train.py) apuntamos al servidor de MLFlow para que sepa cual URL puede usar para hacer el tracking y nombramos el experimento como 'California_Housing'.
Aquí vamos a loggear nuestros parámetros (número de árboles, profundidad máxima de cada árbol, tamaño de entrenamiento, tamaño de test), métricas y el modelo.\
Para entrenar el modelo, usaremos el algoritmo de **Random Forest Regression** que utiliza un conjunto de árboles de decisión para predecir valores continuos. En lugar de depender de un solo árbol de decisión (que puede sobreajustarse fácilmente), el Random Forest Regressor construye muchos árboles de decisión en diferentes subconjuntos de datos y promedia sus resultados para realizar una predicción final.

Por qué usarlo?
- Maneja bien las relaciones no lineales.
- Es resistente al sobreajuste (en comparación con los árboles individuales).
- Puede capturar interacciones complejas entre características.
- Funciona con conjuntos de datos pequeños y grandes.
- Proporciona información sobre la importancia de las características de forma predeterminada.

Cómo funciona?
- Bootstrap Sampling: Cada árbol se entrena con una muestra aleatoria (con reemplazo) de los datos de entrenamiento.
- Feature Randomness: En cada división del árbol, se considera un subconjunto aleatorio de características.
- Aggregation: La predicción final es el promedio de las predicciones de todos los árboles individuales.
Esta aleatoriedad ayuda a reducir la varianza y a construir un modelo más generalizable.

Al final del script, guardamos también el modelo como .pkl local para la API.

## Script de la API

El script de [api.py](https://github.com/adribarra/MLFlow-FastAPI/blob/main/src/api.py) creamos una API minima de aproximadamente 45 líneas que carga el modelo guardado y expone un endpoint de predicción.\
Este script usa el archivo .pkl que se guardó del mejor experimento que hicimos.\
En los valores por default, ponemos los valores medios que obtuvimos en el [EDA](notebooks/CaliforniaHousing_EDA.ipynb) que analizamos del dataset.
