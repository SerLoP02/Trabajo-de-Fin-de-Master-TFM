# Agente de Series Temporales 📈

Este proyecto es el Trabajo de Fin de Máster (TFM) y consiste en el desarrollo de un **Agente de Inteligencia Artificial para el Análisis y Predicción de Series Temporales**, construido utilizando **LangGraph**, **FastAPI** y **Streamlit**. 

El agente está diseñado para ayudar a los usuarios a interactuar con sus datos temporales, permitiendo subir datasets, realizar análisis estadísticos y de estacionariedad, ejecutar transformaciones y entrenar modelos SARIMAX de forma automática e interactiva a través de lenguaje natural.

## 🚀 Características Principales

- **Interfaz Intuitiva (Streamlit):** Carga de archivos CSV, detección automática de formato (delimitador, separador decimal y formato de fechas) e interacción basada en chat con el agente.
- **Backend Robusto (FastAPI):** Arquitectura modular en la que el backend gestiona el estado del agente y atiende las peticiones del cliente (frontend).
- **Agente Inteligente (LangGraph + Google GenAI):** Integración con el modelo `gemini-3.1-flash-lite` (este se puede cambiar en el arhivo `AI_Agent/main.py` línea 47) y un grafo de LangGraph capaz de tomar decisiones, invocar herramientas analíticas y mantener memoria (checkpointing).
- **Herramientas Analíticas Integradas:**
  - **Análisis Estadístico:** Estacionariedad (Test ADF), ruido blanco (Ljung-Box), autocorrelaciones y tests de normalidad.
  - **Visualización:** Gráficos de serie temporal (con detección de outliers), distribuciones (GDF vs KDE), descomposiciones (aditiva/multiplicativa) y gráficas ACF/PACF.
  - **Preprocesamiento:** Imputación de valores nulos y transformaciones matemáticas.
  - **Modelado:** Búsqueda del mejor modelo SARIMAX, entrenamiento iterativo y generación de predicciones (Forecasting).

## 📁 Arquitectura del Proyecto

```text
TFM/
├── AI_Agent/            # Lógica del Agente (LangGraph, Tools de análisis y modelado)
├── Errores/             # Manejo y logs de errores (si aplica)
├── File_Reader/         # Script avanzado para sniffing de CSV (delimitador, fechas)
├── pages/               # Páginas adicionales de Streamlit
├── api.py               # Servidor FastAPI que expone los endpoints del Agente
├── app_con_api.py       # Aplicación principal de Streamlit (Frontend interactivo)
├── utils.py             # Funciones utilitarias para la interfaz, tablas y visualizaciones
├── requirements.txt     # Dependencias del proyecto
├── Dockerfile           # Receta para construir la imagen del contenedor Docker
├── docker-compose.yml   # Definición y orquestación de servicios API y Streamlit
└── README.md            # Documentación
```

## ⚙️ Configuración y Ejecución

Asegúrate de contar con una **API Key de Google (Gemini)**. Crea un archivo `.env` en el directorio raíz:
```env
API_KEY="TU_GEMINI_API_KEY_AQUI"
```

Dispones de dos opciones para iniciar el proyecto:

### Opción 1: Con Docker y Docker Compose (Recomendado 🐳)

Esta opción descarga la versión python usada, instala dependencias y levanta automáticamente ambos servicios (FastAPI y Streamlit) en contenedores aislados con un solo comando.

1. Asegúrate de tener Docker y Docker Compose instalados y en ejecución.
2. Levanta los contenedores ejecutando:
   ```bash
   docker compose up --build
   ```
3. La aplicación estará accesible en:
   - **Streamlit (Frontend):** `http://localhost:8501`
   - **FastAPI (Backend API):** `http://localhost:8000`

### Opción 2: Sin Docker (Ejecución Local Manual)



Para correr el proyecto de forma manual en tu máquina local, necesitarás levantar tanto la API (backend) como Streamlit (frontend) en dos terminales separadas (si sigues esta opción, es recomendable crear un virtual environment):

#### 1. Requisitos de Instalación

El proyecto utiliza Python y diversas librerías de análisis de datos y machine learning. Se recomienda utilizar **Python 3.12** o superior para evitar problemas de compatibilidad.

Para instalar los requisitos, ejecuta:

```bash
pip install -r requirements.txt
```

**Principales Dependencias:**
- `fastapi` y `uvicorn` (Backend API)
- `streamlit` (Frontend)
- `langchain`, `langgraph`, `langchain-google-genai` (Agente de IA)
- `pandas`, `statsmodels`, `pmdarima`, `scipy` (Análisis Numérico)
- `matplotlib` (Visualización)
- `python-dotenv` (Variables de entorno)


#### 2. Iniciar la API de FastAPI
Abre una terminal en la ruta principal del proyecto y ejecuta:
```bash
uvicorn api:api
```
La API estará disponible en `http://127.0.0.1:8000`.

#### 3. Iniciar la Aplicación Streamlit
Abre otra terminal también en la ruta principal del proyecto y ejecuta:
```bash
streamlit run app_con_api.py
```
Streamlit se abrirá automáticamente en tu navegador por defecto mostrando la interfaz del chat.

## 📝 Uso de la Aplicación

1. **Sube tus datos:** Despliega la barra lateral y carga un archivo `.csv`. Asegúrate que la fecha es la **primera columna** y está en formato **strftime**, la serie temporal está en la **segunda columna** y las variables exógenas estén en el resto de columnas
2. **Selecciona la periodicidad:** Semanal, mensual, trimestral u otra, en función de la naturaleza de tu serie temporal.
3. **Interactúa con el Agente:** En la caja de chat, haz preguntas o pídele que analice los datos, por ejemplo:
   - *"Grafica la serie temporal."*
   - *"Haz un test de estacionariedad."*
   - *"Encuentra el mejor modelo SARIMAX."*

---
*Desarrollado como parte del Trabajo de Fin de Máster (TFM).*