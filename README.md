# CV_U2_plant-disease

Pipeline de visión por computador clásica para clasificar imágenes de hojas de PlantVillage según especie y condición sanitaria. El proyecto fue desarrollado para la Unidad 2 del curso CS5362 - Visión por Computador de la Universidad de Ingeniería y Tecnología (UTEC).

La fuente de verdad de este documento es la implementación ejecutada en `new_notebook.ipynb`, los módulos Python y los artefactos generados (`.pkl`, `.csv` y `.png`).

## 1. Descripción del proyecto

El objetivo es estudiar cómo distintas representaciones visuales y clasificadores afectan la clasificación multiclase de enfermedades foliares.

El flujo principal es:

```text
dataset
-> carga de imágenes y etiquetas
-> suavizado Gaussiano
-> segmentación mediante HSV/LAB y Otsu
-> operaciones morfológicas
-> imagen enmascarada
-> extracción de características
-> vectores serializados
-> validación cruzada estratificada
-> evaluación y comparación de modelos
```

La detección de bordes y esquinas también se implementa y visualiza, pero sus resultados no se utilizan como vectores de características en la evaluación final.

## 2. Dataset

El dataset local se encuentra en [plantvillage_dataset](plantvillage_dataset). La celda 6 de [new_notebook.ipynb](new_notebook.ipynb) descubre las carpetas, las ordena alfabéticamente y crea etiquetas one-hot.

En esta copia del proyecto se verificaron:

- 20.638 imágenes válidas.
- 15 clases.
- Una carpeta por clase.
- Imágenes principalmente `.JPG`, con algunos archivos `.png` y `.jpeg`.
- Clases desbalanceadas: la clase más grande tiene 3.209 imágenes y `Potato___healthy` tiene 152.

Las clases y sus índices reales son:

| Índice | Carpeta |
|---:|---|
| 0 | `Pepper__bell___Bacterial_spot` |
| 1 | `Pepper__bell___healthy` |
| 2 | `Potato___Early_blight` |
| 3 | `Potato___Late_blight` |
| 4 | `Potato___healthy` |
| 5 | `Tomato_Bacterial_spot` |
| 6 | `Tomato_Early_blight` |
| 7 | `Tomato_Late_blight` |
| 8 | `Tomato_Leaf_Mold` |
| 9 | `Tomato_Septoria_leaf_spot` |
| 10 | `Tomato_Spider_mites_Two_spotted_spider_mite` |
| 11 | `Tomato__Target_Spot` |
| 12 | `Tomato__Tomato_YellowLeaf__Curl_Virus` |
| 13 | `Tomato__Tomato_mosaic_virus` |
| 14 | `Tomato_healthy` |

El cargador valida cada archivo con `cv2.imread`. En el dataset local existe además un archivo vacío sin extensión dentro de `Tomato__Tomato_YellowLeaf__Curl_Virus`; no se considera una imagen válida.

No se aplica un resize global a todas las imágenes en el pipeline ejecutado. Las dimensiones resultantes de los vectores serializados se indican en la sección de características.

## 3. Estructura del proyecto

```text
CV_U2_plant-disease/
├── new_notebook.ipynb
├── Operations.py
├── PreprocessingPipeline.py
├── class_mappings.txt
├── tabla_resumen_final.csv
├── results_summary.pkl
├── lbp_dataset.pkl
├── hog_features.pkl
├── fourier_features.pkl
├── full_mapping.pkl
├── full_centroids.pkl
├── checkpoints/
├── f1_comparison.png
├── confusion_matrices_grid.png
├── plantvillage_dataset/
└── README.md
```

### Archivos principales

- `new_notebook.ipynb`: experimentación completa, extracción, entrenamiento, validación y visualización.
- `PreprocessingPipeline.py`: suavizado, masking, morfología, bordes y esquinas.
- `Operations.py`: conversiones de color, umbralización y funciones auxiliares de características.
- `class_mappings.txt`: listado de clases usado como referencia textual.
- `tabla_resumen_final.csv`: métricas finales de las 15 combinaciones.
- `results_summary.pkl`: los mismos resultados en formato serializado.
- `checkpoints/*.meta.pkl`: métricas, reportes por clase, matrices de confusión y predicciones out-of-fold.
- `checkpoints/*.model.pkl`: modelos finales reentrenados con todos los datos.

## 4. Metodología

### 4.1 Carga del dataset

La celda 6 crea `DataLoader`. Para cada subcarpeta:

1. Ordena los nombres de las clases.
2. Asigna un índice entero.
3. Valida que OpenCV pueda leer la imagen.
4. Guarda la ruta y una etiqueta one-hot.
5. Aplica el preprocesamiento al leer la imagen cuando se utiliza `dataset_preprocessed`.

Las imágenes se cargan en formato BGR mediante OpenCV.

### 4.2 Preprocesamiento

La configuración utilizada en la celda 10 es:

- Gaussian blur con kernel `3 x 3` y `sigma=0`.
- Canal `S` de HSV, umbralizado con Otsu.
- Canal `A` de LAB, umbralizado con Otsu e invertido.
- Combinación de las máscaras mediante OR lógico.
- Erosión con kernel elíptico `3 x 3`.
- Dilatación con kernel elíptico `10 x 10`.

Estas operaciones están implementadas en `RunPreprocessingOnOne` y `RunMaskingOnOne` de [PreprocessingPipeline.py](PreprocessingPipeline.py), utilizando las conversiones y funciones de [Operations.py](Operations.py).

La salida del masking es la imagen BGR original multiplicada por la máscara. No es únicamente una máscara binaria.

La celda 11 compara la imagen original, la imagen enmascarada sin morfología y la imagen con morfología.

> La operación descrita en el notebook como “apertura” no es una apertura morfológica estándar estricta, porque la erosión y la dilatación usan kernels de tamaños distintos. La documentación conserva el comportamiento real y lo señala como advertencia.

#### Bordes

Las celdas 14 y 15 configuran y visualizan Canny con umbrales `(70, 100)` sobre la imagen enmascarada. `RunEdgeDetectionOnOne` también admite Sobel, pero la configuración ejecutada para esta visualización es Canny.

Los bordes no se incluyen en los vectores usados por los clasificadores finales.

#### Esquinas

Las celdas 17 y 18 comparan Harris y Shi-Tomasi dentro de la región segmentada.

- Shi-Tomasi: máximo de 100 esquinas, `qualityLevel=0.01`, `minDistance=10`.
- Harris: `blockSize=2`, `ksize=3`, `k=0.04`.

La detección de esquinas es exploratoria y visual. No se utiliza como representación evaluada en la tabla de resultados.

### 4.3 Extracción de características

La celda 20 construye `dataset_preprocessed`, que aplica el blur y el masking antes de extraer las características.

#### HOG

Implementado en la celda 24 con `skimage.feature.hog`:

- 9 orientaciones.
- Celdas de `32 x 32` píxeles.
- Bloques de `2 x 2` celdas.
- Imagen convertida a escala de grises.
- Vector final verificado: **1.764 características por imagen**.

HOG representa la distribución local de gradientes y bordes. En este proyecto se utiliza para capturar contornos, nervaduras y estructura espacial de las hojas.

El resultado se guarda en [hog_features.pkl](hog_features.pkl).

#### LBP

Implementado en la celda 22 con `skimage.feature.local_binary_pattern`:

- Radio `R=1`.
- 8 puntos vecinos.
- Método `uniform`.
- Histograma normalizado de 59 bins.
- Vector final verificado: **59 características por imagen**.

LBP representa patrones locales de textura. Puede capturar diferencias de granularidad, manchas y microtexturas asociadas a lesiones.

El resultado se guarda en [lbp_dataset.pkl](lbp_dataset.pkl).

#### Fourier

Implementado en la celda 29 mediante `extract_fourier_features`:

1. Convierte la imagen enmascarada a escala de grises.
2. Divide la imagen, sin resize global, en celdas de `16 x 16`.
3. Calcula una FFT 2D por celda.
4. Desplaza el espectro con `fftshift`.
5. Calcula la magnitud espectral.
6. Resume la energía en 8 bandas radiales.
7. Normaliza el perfil de cada celda.

El vector serializado tiene **2.048 características por imagen**, equivalente a 256 celdas por 8 bandas radiales en el artefacto generado.

Este descriptor representa la distribución de energía en frecuencias espaciales, relacionada con estructura y textura.

El resultado se guarda en [fourier_features.pkl](fourier_features.pkl).

`Operations.py` contiene otra función llamada `extract_custom_fft_concat`, basada en HSV, resize a `128 x 128` y vectores de 1.152 valores. Esa función no es la que generó `fourier_features.pkl` ni la que corresponde a los resultados finales.

#### Bag of Visual Words con ORB

Las celdas 26 y 27 ejecutan la representación BoW utilizada en los experimentos:

- ORB con hasta 500 características por imagen.
- `MiniBatchKMeans` con 256 centroides.
- `batch_size=4096`.
- `max_iter=100`.
- `max_no_improvement=10`.
- `n_init=1`.
- `random_state=42`.
- Histograma final de 256 posiciones.

El resultado se guarda en [full_mapping.pkl](full_mapping.pkl) y el vocabulario en [full_centroids.pkl](full_centroids.pkl), cuya dimensión verificada es `(256, 32)`.

El histograma generado por el notebook contiene conteos de palabras visuales. No se aplica normalización L1 en `BoW_mapToVocabulary`.

Aunque `Operations.py` contiene una función BoW basada en SIFT, la representación evaluada y serializada por el notebook se construyó con **ORB**, no con SIFT.

#### Fusión HOG + LBP

La celda 33 concatena los vectores HOG y LBP:

```text
1764 + 59 = 1823 características
```

Es una fusión temprana por concatenación directa. No se aplicó un ajuste independiente de hiperparámetros para esta combinación; reutiliza los hiperparámetros de HOG, como se indica en la celda 34.

### 4.4 Clasificadores

Se evaluaron KNN, SVM y XGBoost.

#### KNN

Utiliza `weights="distance"`. Los valores de `n_neighbors` y `p` dependen de la representación:

| Representación | `n_neighbors` | `p` |
|---|---:|---:|
| Fourier | 11 | 1 |
| BoW/ORB | 3 | 2 |
| HOG | 3 | 1 |
| LBP | 15 | 2 |
| HOG+LBP | 3 | 1 |

#### SVM

Todas las representaciones utilizan:

```python
kernel="rbf"
gamma="scale"
C=100
```

Para KNN y SVM se aplica `StandardScaler`, ajustado únicamente con el subconjunto de entrenamiento de cada fold.

#### XGBoost

La celda 38 utiliza clasificación multiclase con:

```python
objective="multi:softmax"
tree_method="hist"
device="cuda"
n_jobs=1
eval_metric="mlogloss"
```

Los parámetros varían por representación. La configuración registrada en la celda 34 es:

| Representación | `max_depth` | `learning_rate` | `n_estimators` | `subsample` | `colsample_bytree` | `min_child_weight` |
|---|---:|---:|---:|---:|---:|---:|
| Fourier | 5 | 0.05 | 200 | 0.8 | 1.0 | 1 |
| BoW/ORB | 5 | 0.2 | 300 | 0.8 | 0.8 | 1 |
| HOG | 5 | 0.2 | 300 | 0.8 | 0.8 | 1 |
| LBP | 5 | 0.2 | 300 | 0.8 | 1.0 | 3 |
| HOG+LBP | hereda HOG | hereda HOG | hereda HOG | hereda HOG | hereda HOG | hereda HOG |

### 4.5 Evaluación

La evaluación se implementa en las celdas 34, 37, 38 y 39.

- `StratifiedKFold` de 5 folds.
- `shuffle=True`.
- `random_state=42`.
- `TEST_MODE=False`, por lo que se usaron las 20.638 imágenes.
- No se usa un train/test split independiente en la evaluación final.
- Se calculan accuracy y F1 macro por fold.
- Se generan predicciones out-of-fold para el reporte por clase y la matriz de confusión.
- Después de la validación, cada modelo se reentrena con todo el dataset y se serializa.

Las métricas guardadas son:

- Accuracy media y desviación estándar.
- F1 macro medio y desviación estándar.
- Reporte de clasificación con precision, recall y F1 por clase.
- Matriz de confusión.

## 5. Resultados

Los valores de esta tabla se verificaron directamente en [tabla_resumen_final.csv](tabla_resumen_final.csv), [results_summary.pkl](results_summary.pkl) y los archivos `checkpoints/*.meta.pkl`.

| Características | Clasificador | Accuracy | F1 macro |
|---|---|---:|---:|
| Fourier | KNN | 0.5488 +/- 0.0038 | 0.4859 +/- 0.0045 |
| Fourier | SVM-RBF | 0.6779 +/- 0.0047 | 0.6319 +/- 0.0019 |
| Fourier | XGBoost | 0.6500 +/- 0.0092 | 0.5850 +/- 0.0061 |
| BoW/ORB | KNN | 0.3945 +/- 0.0070 | 0.3336 +/- 0.0084 |
| BoW/ORB | SVM-RBF | 0.5998 +/- 0.0037 | 0.5505 +/- 0.0029 |
| BoW/ORB | XGBoost | 0.5711 +/- 0.0076 | 0.5151 +/- 0.0075 |
| HOG | KNN | 0.6702 +/- 0.0053 | 0.6183 +/- 0.0057 |
| HOG | SVM-RBF | 0.8121 +/- 0.0047 | 0.7821 +/- 0.0062 |
| HOG | XGBoost | 0.7641 +/- 0.0040 | 0.7120 +/- 0.0074 |
| LBP | KNN | 0.5547 +/- 0.0078 | 0.4822 +/- 0.0069 |
| LBP | SVM-RBF | 0.6310 +/- 0.0068 | 0.5707 +/- 0.0082 |
| LBP | XGBoost | 0.5878 +/- 0.0051 | 0.5258 +/- 0.0049 |
| HOG+LBP | KNN | 0.6742 +/- 0.0062 | 0.6221 +/- 0.0066 |
| HOG+LBP | SVM-RBF | **0.8228 +/- 0.0044** | **0.7933 +/- 0.0047** |
| HOG+LBP | XGBoost | 0.8023 +/- 0.0064 | 0.7612 +/- 0.0097 |

## 6. Análisis de resultados

### Mejor y peor combinación

**Resultado observado:** HOG+LBP con SVM-RBF fue la mejor combinación, con accuracy `0.8228` y F1 macro `0.7933`.

**Resultado observado:** BoW/ORB con KNN fue la peor combinación, con accuracy `0.3945` y F1 macro `0.3336`.

**Interpretación:** HOG aporta estructura y gradientes, mientras LBP aporta textura local. La combinación ofrece información complementaria. Esta explicación es una hipótesis compatible con el resultado, no una prueba causal independiente.

### Comportamiento de las representaciones

- **HOG:** fue el mejor descriptor individual. Con SVM obtuvo F1 macro `0.7821`.
- **LBP:** obtuvo resultados inferiores a HOG, lo que indica que la textura local por sí sola no fue suficiente para separar todas las clases.
- **Fourier:** con SVM obtuvo F1 macro `0.6319`, superior a LBP con KNN y SVM en algunas configuraciones, pero inferior a HOG.
- **BoW/ORB:** fue la representación más débil en esta evaluación, especialmente con KNN.
- **HOG+LBP:** mejoró HOG con SVM de `0.7821` a `0.7933` de F1 macro.

### Comportamiento de los clasificadores

**Resultado observado:** SVM-RBF fue el mejor clasificador para las cinco representaciones.

**Resultado observado:** XGBoost fue generalmente segundo, mientras KNN obtuvo los resultados más bajos o cercanos a los más bajos.

**Interpretación:** las fronteras no lineales de SVM parecen adaptarse mejor a los vectores de características utilizados. KNN puede verse perjudicado por la dimensionalidad y por la distribución de distancias entre imágenes.

### Accuracy frente a F1 macro

Accuracy es mayor que F1 macro en todas las combinaciones. Esto es compatible con el desbalance del dataset: las clases con más imágenes pueden contribuir más a accuracy, mientras F1 macro pondera todas las clases por igual.

### Matrices de confusión

Las matrices normalizadas se encuentran en [confusion_matrices_grid.png](confusion_matrices_grid.png). Para el mejor modelo, las clases con menor F1 fueron:

- `Tomato_Early_blight`: `0.5371`.
- `Potato___healthy`: `0.6320`.
- `Tomato_Septoria_leaf_spot`: `0.6990`.
- `Tomato__Target_Spot`: `0.7487`.
- `Tomato_Late_blight`: `0.7633`.

Las confusiones más frecuentes del mejor modelo fueron entre `Tomato_Early_blight` y `Tomato_Late_blight`, y entre `Tomato_Septoria_leaf_spot`, `Tomato_Late_blight` y `Tomato_Bacterial_spot`.

## 7. Conclusiones

1. HOG fue la mejor representación individual de las evaluadas.
2. La concatenación HOG+LBP produjo el mejor resultado global.
3. SVM con kernel RBF fue el clasificador más eficaz en las cinco representaciones.
4. Las características de textura de LBP aportaron una mejora pequeña al combinarse con HOG.
5. BoW/ORB fue la representación con peor rendimiento en esta configuración.
6. Las clases de tomate con síntomas visualmente parecidos fueron las más difíciles de distinguir.
7. Los resultados corresponden a validación cruzada sobre PlantVillage y no permiten asegurar el mismo rendimiento en imágenes tomadas en condiciones reales.

## 8. Advertencias metodológicas

Estas observaciones no se modificaron porque el objetivo de esta actualización fue documentar la implementación existente.

### ⚠ POSIBLE PROBLEMA EN IMPLEMENTACIÓN: vocabulario BoW

El vocabulario ORB se construye con descriptores de todas las imágenes antes de ejecutar `StratifiedKFold`. Aunque no se utilizan las etiquetas para crear los centroides, la información visual de los folds de validación participa en la construcción del vocabulario. Una evaluación estricta debería construir el vocabulario usando únicamente el fold de entrenamiento dentro de cada partición.

### ⚠ POSIBLE PROBLEMA EN IMPLEMENTACIÓN: diferencias entre módulos

`Operations.py` contiene funciones alternativas basadas en SIFT y otra versión de Fourier, pero los resultados finales documentados aquí provienen de las celdas específicas del notebook que usan ORB y `extract_fourier_features`.

### ⚠ POSIBLE PROBLEMA EN IMPLEMENTACIÓN: morfología

La erosión `3 x 3` seguida de la dilatación `10 x 10` se describe informalmente como apertura, pero no corresponde a una apertura estándar con un único kernel.

### ⚠ POSIBLE PROBLEMA EN IMPLEMENTACIÓN: nombres de clases

La lista fija utilizada para rotular las figuras contiene algunos nombres distintos de los nombres físicos de las carpetas, por ejemplo `Tomato__Bacterial_spot` frente a `Tomato_Bacterial_spot`. Los índices numéricos se mantienen en el mismo orden, pero los textos no son idénticos.

## 9. Archivos generados

### Características

- [lbp_dataset.pkl](lbp_dataset.pkl): 20.638 vectores de 59 posiciones.
- [hog_features.pkl](hog_features.pkl): 20.638 vectores de 1.764 posiciones.
- [fourier_features.pkl](fourier_features.pkl): 20.638 vectores de 2.048 posiciones.
- [full_mapping.pkl](full_mapping.pkl): 20.638 histogramas BoW de 256 posiciones.
- [full_centroids.pkl](full_centroids.pkl): vocabulario ORB de dimensión `(256, 32)`.

### Resultados y modelos

- [results_summary.pkl](results_summary.pkl): resumen serializado de las 15 combinaciones.
- [tabla_resumen_final.csv](tabla_resumen_final.csv): tabla de métricas.
- `checkpoints/`: modelos finales y metadatos por representación y clasificador.

### Figuras

- [f1_comparison.png](f1_comparison.png): comparación de F1 macro con barras de desviación estándar.
- [confusion_matrices_grid.png](confusion_matrices_grid.png): matrices de confusión normalizadas para las 15 combinaciones.

## 10. Requisitos y ejecución

El notebook importa Python, NumPy, OpenCV, Matplotlib, scikit-image, scikit-learn, SciPy, Joblib, XGBoost y KaggleHub.

La primera celda de instalación del notebook instala principalmente:

```bash
pip install opencv-python kaggle kagglehub matplotlib scikit-learn scikit-image
```

El notebook también requiere las dependencias importadas en la celda 4, incluida `numpy`, `scipy`, `joblib` y `xgboost`.

Para reproducir el flujo se debe ejecutar [new_notebook.ipynb](new_notebook.ipynb) en orden:

1. Instalar dependencias y cargar módulos.
2. Cargar y validar el dataset.
3. Visualizar masking, bordes y esquinas.
4. Extraer o cargar las características serializadas.
5. Permutar y alinear las representaciones.
6. Entrenar y evaluar KNN, SVM y XGBoost.
7. Generar checkpoints, tabla y figuras.

La descarga desde Kaggle solo se intenta si no existe `plantvillage_dataset` localmente.

## 11. Autores

- Thiago Frias Pinto
- Chiara Olenka Lazaro Condor
- Angel Obed Mora Huamanchay
- Juan Diego Prochazka Zegarra
- Mishelle Stephany Villarreal Falcón

**Universidad de Ingeniería y Tecnología (UTEC)**  
**Curso:** CS5362 - Visión por Computador  
**Ciclo académico:** 2026-2

## 12. Repositorio

https://github.com/MishelleVF/CV_U2_plant-disease
