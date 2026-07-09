# 🎥 Vigia: Sistema de Detección de Eventos de Seguridad y Hurto con MMAction2

**Vigía** es una plataforma inteligente de análisis y clasificación de video orientada a la seguridad. Utilizando el framework de vanguardia **OpenMMLab MMAction2**, Vigía procesa videos de seguridad para clasificar automáticamente actividades y alertar sobre conductas delictivas (hurto), situaciones de violencia/riesgo, o confirmar actividades normales.

---

## 📊 Descripción del Modelo y Arquitectura

El sistema de clasificación de video está basado en el algoritmo **TSN (Temporal Segment Network)**, el cual divide los videos en segmentos temporales uniformes para modelar tanto la información estática (espacial) como la dinámica (temporal).

### Especificaciones Técnicas del Modelo:
* **Framework:** MMAction2 v1.2.0 (OpenMMLab).
* **Backbone (Extractor de Características):** **ResNet-50** preentrenado con ImageNet.
* **Algoritmo:** Temporal Segment Network (TSN).
* **Entrada de Datos:** Muestreo temporal uniforme de 8 segmentos por clip (`num_clips=8`), decodificado a través del motor `Decord` y procesado mediante recortes multiescala a una resolución final de 224x224 píxeles.
* **Número de Clases de Salida:** 21 clases.

---

## 📁 Clasificación del Conjunto de Datos (Dataset)

El modelo está entrenado para clasificar comportamientos en **21 clases distintas**, las cuales se agrupan en tres niveles de alerta crítica en el script de interpretación:

| Alerta | Descripción | Clases Incluidas |
| :--- | :--- | :--- |
| 🚨 **Hurto (Theft)** | Comportamientos delictivos de apropiación indebida. | `Stealing`, `Shoplifting`, `Robbery`, `Burglary` |
| 🔥 **Riesgo / Violencia** | Situaciones peligrosas, comportamiento agresivo o altercados. | `Fighting`, `Assault`, `Abuse`, `Vandalism`, `Shooting`, `Explosion`, `Arson`, `Arrest`, `Roadaccidents` |
| ✅ **Actividad Normal** | Actividades cotidianas o inofensivas. | `Normal`, `Walking`, `Walking_While_Using_Phone`, `Walking_While_Reading_Book`, `Standing_Still`, `Sitting`, `Clapping`, `Meet_and_Split` |

---

## 📈 Detalles de Entrenamiento

* **Ciclos de Entrenamiento:** Configurado para un máximo de 100 épocas (`max_epochs=100`) a través del loop basado en épocas de MMEngine (`EpochBasedTrainLoop`).
* **Optimización y Validación:** Validación programada a intervalos de 1 época para monitorear el desempeño frente a sobreajustes.
* **Mejor Checkpoint:** **`best_acc_top1_epoch_13.pth`** (Época 13), logrando el pico más alto de exactitud (Accuracy Top-1) con un peso optimizado de **90.9 MB**.

---

## 💻 Ejecución Local (Windows / Linux)

### Paso 1: Configurar el Entorno
Hemos creado un script que automatiza la creación del entorno Conda compatible en tu máquina local:
1. Abre tu consola (CMD o PowerShell) en la raíz del proyecto.
2. Ejecuta el script de instalación:
   ```cmd
   setup_env.bat
   ```
   *Esto creará el entorno `mmaction2` bajo Python 3.9 e instalará PyTorch 2.1.2 (CPU), torchvision, openmim, mmengine y mmcv-2.1.0 de forma automática.*

### Paso 2: Ejecutar el Análisis de Video
Con el entorno activo, ejecuta la inferencia con el siguiente comando:
```bash
python detectar_evento.py ruta/a/tu/video.mp4 --checkpoint work_dirs/tsn_hurto/best_acc_top1_epoch_13.pth
```

El script imprimirá una lectura simplificada tanto para ingenieros como para operadores de seguridad:
* Evento más probable detectado y confianza del modelo.
* Top 5 posibilidades con sus porcentajes individuales.
* Lectura no técnica clara (ej. *ALERTA ALTA*, *ALERTA DE RIESGO*, *SIN ALERTA*).
* Probabilidad agregada para cada una de las 3 categorías principales.

---

## ☁️ Despliegue en la Nube (Online)

### Opción 1: Aplicación Web en Streamlit Community Cloud (Gratuito)
Puedes desplegar este proyecto en la nube para que funcione con una interfaz web interactiva a través de **Streamlit Community Cloud** (100% gratis):

1. Sube este repositorio a tu GitHub (incluyendo las carpetas `mmaction`, `configs`, `demo` y el checkpoint en `work_dirs/tsn_hurto/best_acc_top1_epoch_13.pth`).
2. El repositorio ya cuenta con los archivos de configuración requeridos para la nube:
   * **`streamlit_app.py`:** El script que levanta la interfaz gráfica del detector de video.
   * **`packages.txt`:** Lista de dependencias del sistema operativo Debian Linux del servidor (`ffmpeg`, `libgl1`, `libglx-mesa0` para video y gráficos).
   * **`runtime.txt`:** Fuerza al servidor a inicializarse con la versión compatible **`python-3.9`**.
   * **`requirements.txt`:** Listado plano de dependencias de Python optimizadas para correr la inferencia en CPU.
3. Ve a [Streamlit Share](https://share.streamlit.io/), conecta tu repositorio, apunta a `streamlit_app.py` y haz clic en **Deploy**.

### Opción 2: Jupyter Notebook en Google Colab / Kaggle (GPU Gratuita)
Si necesitas procesar lotes grandes de video usando aceleración gráfica por GPU:
1. Sube el archivo **`detectar_evento_notebook.ipynb`** (ubicado en la raíz del proyecto) a Google Colab o Kaggle.
2. Activa el entorno de ejecución por GPU (T4 GPU).
3. Ejecuta las celdas ordenadamente. El notebook se encargará de instalar las dependencias en el servidor temporal y correrá el detector usando el acelerador de hardware de CUDA (`--device cuda`).
