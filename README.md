# MLFlow-FastAPI

El objetivo de este challenge es entrenar un modelo y utilizar las capacidades de MLFlow para loggear las características de las ejecuciones de todos los experimentos que ejecute, así como el modelo entrenado, y utilizar los resultados para decidir cómo ir variando el valor de los parámetros de entrenamiento del algoritmo para ir mejorando las métricas de funcionamiento y que a la vez comparar el funcionamiento hasta decidir cuál es el que mejor funcionamiento ofrece.

El proceso de este challenge se dividirá en los siguientes pasos:
- [Paso 0](#paso-0): Preparación del entorno:
  - Instalación de dependencias, creación de la carpeta del proyecto, inicialización de Git y DVC y arrancar MLFlow.
- Paso 1: Datos:
  - Descargar el dataset y versionarlo con DVC.
- Paso 2: Entrenamiento con MLFlow:
  - Creación de script con logging manual de parámetros y métricas. 
- Paso 3: Experimentos:
  - Lanzar 5 experimentos variando el parámetro necesario. Comparar en MLFlow y decidir a partir de los resultados.
- Cierre: Puesta en producción:
  - Crear el script de la API que carga el modelo y expone el endpoint de predicción. Arrancar con uvicorn.
 
## Dataset
Usaremos el dataset de California Housing incluido en scikit-learn. Tiene 20,640 filas y 8 variables numericas (ingreso medio, edad de la vivienda, habitaciones, etc.) para predecir el precio medio de las casas. No necesita ninguna descarga externa. 
En este Jupyter Notebook está el EDA que realicé para revisar su estructura y confirmar factores extras como distribución o correlaciones importantes. 

## Paso 0
Esta práctica asume que ya se tiene instalado WSL2 con Ubuntu y Miniconda. Si no se han completado esos pasos, en [este repositorio](github.com/adribarra/MLOpsMioti) están las instrucciones 

