# Tarea-1---MII910-Deep-Learning---PUCV
__Pontificia Universidad Católica de Valparaíso — Escuela de Ingeniería Informática__ 
__Semestre 1-2026 · Profesor: Carlos Valle__

### __Integrantes: Nicolas Zarate, David Caceres, Milenka Zuvic.__


# Parte 1 de la Tarea - Clasificación de Niveles de Obesidad

## Descripción

Implementación de redes neuronales densas (Feed Forward) en TensorFlow/Keras para clasificar niveles de obesidad a partir de atributos físicos y hábitos alimenticios, utilizando el dataset [Estimation of Obesity Levels](https://archive.ics.uci.edu/dataset/544) de UCI Machine Learning Repository.

El proyecto abarca desde el análisis exploratorio hasta la optimización de hiperparámetros mediante búsqueda aleatoria y estudios de ablación/sensibilidad.

## Dataset

| Atributo | Descripción |
|----------|-------------|
| `Gender` | Sexo biológico |
| `Age` | Edad en años |
| `Height` | Estatura (m) |
| `Weight` | Peso corporal (kg) |
| `family_history_with_overweight` | Historial familiar de sobrepeso |
| `FAVC` | Consumo de alimentos altos en calorías |
| `FCVC` | Frecuencia de consumo de verduras |
| `NCP` | Número de comidas principales |
| `CAEC` | Consumo entre comidas |
| `SMOKE` | Fumador |
| `CH2O` | Consumo diario de agua |
| `SCC` | Monitoreo calórico |
| `FAF` | Frecuencia de actividad física |
| `TUE` | Tiempo de uso de tecnología |
| `CALC` | Consumo de alcohol |
| `MTRANS` | Medio de transporte principal |
| **`NObeyesdad`** | **Variable objetivo — 7 niveles de obesidad** |

## Estructura del Notebook

### Parte 1 — Conceptos Básicos de Redes Neuronales (50%)

**1.a · Carga y preprocesamiento**
- Análisis exploratorio con estadísticas descriptivas, boxplots e histogramas
- Boxplots desglosados por nivel de obesidad y género
- Codificación de variables categóricas con `LabelEncoder`
- Split estratificado 70/20/10 (train/val/test)
- Estandarización con `StandardScaler`

**1.b · Función de evaluación**
- Métrica principal: F1-score macro sobre validación

**1.c · Búsqueda de hiperparámetros**
- Random search con 20 trials sobre activación, profundidad, neuronas y learning rate
- Mejor configuración: `tanh`, 1 capa, 100 neuronas, SGD con lr=0.1

**1.d · Estudios de ablación/sensibilidad**
- Profundidad de la red (1–6 capas)
- Neuronas por capa (10–200)
- Tasa de aprendizaje (escala logarítmica)
- Batch size (16–256)
- Optimizadores (SGD, Adam, RMSprop, Adagrad, Adadelta)
- Inicialización de pesos (Glorot, He, random, zeros)

**1.e · Técnicas de regularización**
- Regularización L1 y L2 con distintos valores de λ
- Dropout (0.0–0.6)
- Early Stopping con distintos valores de paciencia

## Resultados Principales

| Métrica | Valor |
|---------|-------|
| F1-score validación | **0.9856** |
| F1-score test | **0.9755** |
| Gap val–test | 0.01 |

La configuración óptima combina activación `tanh`, una sola capa oculta de 100 neuronas, optimizador SGD con learning rate 0.1, inicializador `he_uniform` y dropout de 0.4. La brecha mínima entre validación y test indica buena generalización sin sobreajuste.

## Stack Tecnológico

- Python 3.x
- TensorFlow / Keras
- Pandas · NumPy
- Matplotlib · Seaborn
- Scikit-learn

## Cómo Ejecutar

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/tarea1-redes-neuronales.git

# Instalar dependencias
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn ucimlrepo

# Ejecutar el notebook
jupyter notebook Tarea_1_PARTE_1_FINAL.ipynb
```

> **Nota:** Se recomienda ejecutar en Google Colab con GPU/TPU como acelerador para reducir tiempos de entrenamiento.

# Parte 2: Desafío Kaggle. Clasificación de Géneros Músicales

Clasificación automática de audio en 8 géneros musicales mediante un ensemble de 6 modelos (FFNN×3 + LightGBM×3) sobre embeddings preentrenados extraídos de tres segmentos temporales independientes.

**Score público Kaggle:** v13 = 0.609 | v14 = TBD  
**OOF Macro F1:** > 0.703

---

## Problema

Dado un archivo MP3, clasificarlo en uno de 8 géneros: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop o Rock. Métrica de evaluación: **Macro F1-Score**.

## Dataset

- **Train:** 6,267 archivos MP3 (3 corruptos excluidos)
- **Test:** 800 archivos MP3
- **Clases:** 8 géneros con leve desbalance (Hip-Hop/Electronic más representados, Folk menos)
- **Duración:** tracks recortados a 30 segundos, procesados en ventanas de 10s

## Arquitectura

```
Audio MP3 (30s)
    │
    ├── first10 (0–10s) ──→ AST(768D) + PANNs(2048D) ──→ L2-norm ──→ [2816D] ──→ FFNN + LGB
    ├── mid10   (D/2±5s) ──→ AST(768D) + PANNs(2048D) ──→ L2-norm ──→ [2816D] ──→ FFNN + LGB
    └── last10  (-10s)   ──→ AST(768D) + PANNs(2048D) ──→ L2-norm ──→ [2816D] ──→ FFNN + LGB
                                                                            │
                                                                     Blend 6 modelos
                                                                            │
                                                                      Predicción final
```

### Extractores de features (frozen)

| Modelo | Arquitectura | Dim. embedding | Sample rate | Preentrenado en |
|--------|-------------|----------------|-------------|-----------------|
| AST | Transformer | 768 | 16 kHz | AudioSet |
| PANNs/Cnn14 | CNN | 2,048 | 32 kHz | AudioSet |


### Clasificador FFNN (MusicFFNN)

```
Input(2816) → [Linear → BatchNorm → ReLU → Dropout] × N bloques → Linear(8)
```

- **Manifold Mixup:** mezcla representaciones en capas intermedias aleatorias (λ ~ Beta(α,α))
- **Focal Loss:** (1-pt)^γ × CE con label smoothing — concentra gradientes en ejemplos difíciles
- **Entrenamiento:** Adam + ReduceLROnPlateau, early stopping por Macro F1 en validación

### LightGBM (modelo paralelo)

Entrenado directamente sobre los supervectores de 2816D por cache. Complementa al FFNN: la red neuronal aprende fronteras suaves, LGB aprende reglas de corte.

## Búsqueda de hiperparámetros

**Estrategia:** OFAT (One-Factor-At-a-Time) independiente por cache, 3-fold × seed=42.

| Parámetro | Valores explorados | Baseline |
|-----------|-------------------|----------|
| dropout | 0.2, 0.3, 0.4, 0.5 | 0.4 |
| focal_gamma | 1.0, 2.0, 3.0, 4.0 | 3.0 |
| label_smoothing | 0.0, 0.05, 0.10, 0.15 | 0.1 |
| lr | 5e-4, 1e-3, 2e-3, 5e-3 | 1e-3 |
| weight_decay | 0.0, 1e-5, 1e-4, 1e-3 | 1e-4 |
| hidden_dims | [256,128], [512,256], [768,384], [512,256,128] | [512,256] |
| mixup_alpha | 0.1, 0.2, 0.4 | 0.2 |
| batch_size | 16, 32, 64 | 32 |

**Run final:** 5-fold stratified × 5 seeds = 25 modelos por cache × 3 caches = 75 FFNNs + 15 LGBs = **90 modelos totales**.

## Ensemble

Blend de 6 modelos con 3 configuraciones fijas de pesos (sin optimización continua para evitar overfitting OOF):

| Config | Descripción |
|--------|-------------|
| Equal | 1/6 por modelo |
| OOF-prop | Proporcional al F1 individual de cada modelo |
| FFNN-heavy | ~18.3% FFNN, ~15% LGB por cache |

## Evolución del proyecto

| Versión | Public F1 | Cambio clave |
|---------|-----------|-------------|
| v0 | ~0.40 (val) | Features clásicas artesanales + sklearn |
| v0.5 | ~0.495 | Features tabulares + MLP ensemble |
| v1 | 0.554 | FFNN + LGB + AST + PANNs |
| v6 | 0.577 | Focal Loss + Label Smoothing |
| v8 | 0.586 | Manifold Mixup + LGB 4° modelo |
| v13 | 0.609 | Dual cache (first10 + last10) |
| v14 | TBD | Triple cache + OFAT por cache + 5 seeds |

## Requisitos

```
torch
transformers
librosa
lightgbm
scikit-learn
numpy
pandas
matplotlib
seaborn
```
## Resultados por clase (OOF v14)

| Género | F1 | Observación |
|--------|-----|-------------|
| Hip-Hop | 0.834 | Mejor clase — firma rítmica distintiva |
| International | 0.812 | Texturas vocales e instrumentales únicas |
| Electronic | 0.732 | Patrones rítmicos y síntesis reconocibles |
| Folk | 0.732 | Armónicos acústicos dispersos |
| Rock | 0.726 | Confusión con Pop |
| Instrumental | 0.691 | Baja amplitud, diverso |
| Experimental | 0.627 | Inherentemente heterogéneo |
| Pop | 0.508 | Peor clase — "atractor difuso" de errores |

## Lecciones clave

- **Embeddings preentrenados > features artesanales:** AST+PANNs superan MFCC/chroma/mel desde v1
- **Diversidad temporal es la palanca más efectiva:** el salto v8→v13 (+0.023) fue el mayor del proyecto
- **No optimizar pesos de blend sobre OOF:** produce overfitting que no transfiere (lección v2/v6)
- **Manifold Mixup > Input Mixup:** regularizar en el espacio latente es más efectivo
- **CLAP no mejoró sobre AST+PANNs:** más dimensiones ≠ mejor (v10)






