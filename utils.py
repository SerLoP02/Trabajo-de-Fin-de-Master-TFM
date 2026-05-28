import streamlit as st


import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


import scipy.stats as sct
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


########################## SELECCIONAR PERIODICIDAD ##########################
########################## SELECCIONAR PERIODICIDAD ##########################
def seleccionar_periodicidad() -> int:

    opciones_period = ["Selecciona...", "Semanal", "Mensual", "Trimestral", "Otra", "Ninguna"]

    periodicidad = st.selectbox(
        "Selecciona una", 
        opciones_period,
    )

    if periodicidad == "Selecciona...":
        st.info("Por favor, selecciona la periodicidad de la serie para continuar.")
        st.stop() 
    elif periodicidad == "Otra":
        periodicidad = st.number_input(
            "Introduce la cantidad exacta de periodos:",  
            min_value = 1,
            value = 1,
            max_value = 365  
        )
        st.success(f"Has seleccionado una periodicidad: {periodicidad}")
    else:
        st.success(f"Has seleccionado una periodicidad: {periodicidad}")
 
    match periodicidad:
        case "Semanal": periodicidad = 7
        case "Mensual": periodicidad = 12
        case "Trimestral": periodicidad = 4
        case "Ninguna": periodicidad = 1
        
    return periodicidad


########################## FORMATEAR LA RESPUESTA ##########################
########################## FORMATEAR LA RESPUESTA ##########################
def extraer_texto_limpio(mensaje):
    
    # Si el contenido es una lista
    if isinstance(mensaje.content, list):
        textos = []
        for bloque in mensaje.content:
            if isinstance(bloque, dict) and bloque.get("type") == "text":
                textos.append(bloque.get("text"))
        return "\n".join(textos)
    
    # Si el contenido ya es un string normal
    return mensaje.content

def extraer_texto_limpio_con_api(mensaje):

    contenido = mensaje.get("content", "")
    if isinstance(contenido, list):
        textos = []
        for bloque in contenido:
            if isinstance(bloque, dict) and bloque.get("type") == "text":
                textos.append(bloque.get("text"))
        return "\n".join(textos)
    return contenido

########################## PLOT SERIE TEMPORAL ##########################
########################## PLOT SERIE TEMPORAL ##########################
def plotear_serie(df: pd.DataFrame) -> None:

    try:

        df = df.dropna()
        col_names = df.columns
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
        
        mean = y.mean()
        std = y.std()
        
        limite_superior = mean + (3 * std)
        limite_inferior = mean - (3 * std)
        
        outliers = df[(y > limite_superior) | (y < limite_inferior)]

        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(x, y, label="Serie Temporal", alpha=0.8)
        
        ax.axhline(mean, linestyle="-", c="black", alpha=0.7, label="Media")
        ax.axhline(limite_superior, linestyle="--", c="green", label="+3 σ")
        ax.axhline(limite_inferior, linestyle="--", c="green", label="- 3 σ")
        
        if not outliers.empty:
            ax.scatter(
                outliers.iloc[:, 0], 
                outliers.iloc[:, 1], 
                color="red", 
                zorder=2, 
                s=12,
                label=f"Outliers ({len(outliers)})"
            )
            
        ax.set_xlabel(col_names[0])
        ax.set_ylabel(col_names[1])
        ax.set_title("Serie Temporal")
        
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error("Se ha producido un error al graficar la serie" + str(e))


########################## TABLA CON VALOES NULOS ##########################
########################## TABLA CON VALOES NULOS ##########################
def valores_nulos(df: pd.DataFrame) -> None:

    st.markdown("""
    <style>
        /* 1. Make the table lines solid and dark */
        table {
            border: 1.5px solid #262730 !important;
            color: #000000 !important; /* Forces text to pure black */
        }
        
        /* 2. Style headers: Bold, dark, and slightly shaded background */
        th {
            background-color: #f0f2f6 !important;
            color: #000000 !important;
            font-weight: bold !important;
            border: 1px solid #262730 !important;
        }
        
        /* 3. Style data cells: Make text heavy and borders clear */
        td {
            border: 1px solid #262730 !important;
            color: #000000 !important;
            font-weight: 500 !important; /* Medium-bold for better legibility */
        }
    </style>
    """, unsafe_allow_html=True)

    st.table(df.isnull().sum().rename("Valores Nulos"), width="content")



########################## TABLA CON ESTADÍSTICOS ##########################
########################## TABLA CON ESTADÍSTICOS ##########################
def calculo_estadísticos(df: pd.DataFrame) -> None:
    
    st.markdown("""
        <style>
            /* 1. Make the table lines solid and dark */
            table {
                border: 1.5px solid #262730 !important;
                color: #000000 !important; /* Forces text to pure black */
            }
            
            /* 2. Style headers: Bold, dark, and slightly shaded background */
            th {
                background-color: #f0f2f6 !important;
                color: #000000 !important;
                font-weight: bold !important;
                border: 1px solid #262730 !important;
            }
            
            /* 3. Style data cells: Make text heavy and borders clear */
            td {
                border: 1px solid #262730 !important;
                color: #000000 !important;
                font-weight: 500 !important; /* Medium-bold for better legibility */
            }
        </style>
    """, unsafe_allow_html=True)

    y = df.iloc[:, 1]

    stats = {
        "Nº Observaciones Totales": f"{len(y)}", 
        "Media": f"{y.mean():.3f}",
        "Mediana": f"{y.median():.3f}",
        "Desviación Estándar": f"{y.std():.3f}",
        "Mínimo": f"{y.min():.3f}",
        "Máximo": f"{y.max():.3f}",
        "Asimetría (Skewness)": f"{y.skew():.3f}",
        "Curtosis (Kurtosis)": f"{y.kurt():.3f}",
    }

    df_stats = pd.DataFrame(list(stats.items()), columns=["Estadístico", "Valor"])    
    df_stats = df_stats.set_index("Estadístico")

    st.table(df_stats, width="content") 


########################## ANÁLISIS DE SERIE TEMPORAL ##########################
########################## ANÁLISIS DE SERIE TEMPORAL ##########################
def mostrar_resultados_serie(resultado_dict: dict) -> None:
    """
    Toma el diccionario de resultados de time_series_analysis 
    y lo dibuja de forma elegante en Streamlit.
    """
    
    # Extraer y aplanar los datos numéricos/booleanos para la tabla
    datos_tabla = [
        {
            "Prueba / Característica": "Estacionariedad (Test ADF)",
            "Cumple": "✅ Sí" if resultado_dict["Estacionaridad"]["Es estacionaria"] else "❌ No",
            "P-Valor": f"{resultado_dict['Estacionaridad']['ADF pvalor']:.4f}"
        },
        {
            "Prueba / Característica": "Media Cero (Newey-West OLS)",
            "Cumple": "✅ Sí" if resultado_dict["Ruido Blanco"]["Media zero"]["Tiene media cero"] else "❌ No",
            "P-Valor": f"{resultado_dict['Ruido Blanco']['Media zero']['Newey-West-OLS pvalor']:.4f}"
        },
        {
            "Prueba / Característica": "Sin Autocorrelación Lineal (Ljung-Box)",
            "Cumple": "✅ Sí" if resultado_dict["Ruido Blanco"]["No Autocorrelacion Lineal"]["No Autocorrelacion Lineal"] else "❌ No",
            "P-Valor": f"{resultado_dict['Ruido Blanco']['No Autocorrelacion Lineal']['Ljung-Box pvalor']:.4f}"
        },
        {
            "Prueba / Característica": "⭐ ¿Es Ruido Blanco Estándar?",
            "Cumple": "✅ Sí" if resultado_dict["Ruido Blanco"]["Es Ruido Blanco"] else "❌ No",
            "P-Valor": "-"
        }
    ]
    
    df_tabla = pd.DataFrame(datos_tabla)
    
    # Dibujar la tabla
    st.subheader("📊 Resumen Estadístico")
    st.dataframe(
        df_tabla, 
        hide_index=True, 
    )
    
    # Extraer y mostrar las conclusiones de texto
    st.subheader("🧠 Conclusiones Avanzadas")
    
    info_estricto = resultado_dict["Ruido Blanco Estricto"]["Informacion"]
    info_gauss = resultado_dict["Ruido Blanco Gaussiano"]["Información"]
    
    # Usamos st.warning (amarillo) si la serie falla, y st.info (azul) o st.success (verde) si pasa
    if "no puede ser" in info_estricto or "La serie no es" in info_estricto:
        st.warning(f"**Ruido Blanco Estricto:** {info_estricto}")
    else:
        st.success(f"**Ruido Blanco Estricto:** {info_estricto}")
        
    if "no puede ser" in info_gauss or "La serie no es" in info_gauss:
        st.warning(f"**Ruido Blanco Gaussiano:** {info_gauss}")
    else:
        st.success(f"**Ruido Blanco Gaussiano:** {info_gauss}")

########################## PLOT FUNCIÓN DISTRIBUCIÓN ##########################
########################## PLOT FUNCIÓN DISTRIBUCIÓN ##########################
def plotear_distribucion(df: pd.DataFrame) -> None:

    try:    
        y = df.iloc[:, 1]
        
        mu = y.mean()
        sigma = y.std()

        x = np.linspace(mu - 5*sigma, mu + 5*sigma, 1000)

        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Distribución Gaussiana Teórica
        pdf = sct.norm.pdf(x, mu, sigma)
        ax.plot(x, pdf, lw=2, color="g", label="GDF Teórica")

        # Histograma
        ax.hist(y, density=True, range=(mu - 3*sigma, mu + 3*sigma), color="r", alpha=0.5, label="Histograma")

        # KDE
        kde = sct.gaussian_kde(y)
        ax.plot(x, kde(x), lw=2, color="blue", label="KDE")

        ax.set_ylabel("Densidad")
        ax.set_xlim(mu - 5*sigma, mu + 5*sigma)
        ax.legend()
        
        st.pyplot(fig)
        plt.close(fig) 

    except Exception as e:
        st.error("Se ha producido un problema al calcular la distribución")

########################## PLOT AUTOCORRELACIONES ##########################
########################## PLOT AUTOCORRELACIONES ##########################
def plotear_autocorrealciones(df: pd.DataFrame) -> None:

    try: 
        num_lags = 15
        y = df.iloc[:, 1]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        plot_acf(y, ax=ax[0], lags=num_lags, title="Autocorrelación")
        plot_pacf(y, ax=ax[1], lags=num_lags, method="ols", title="Autocorrelación parcial")  

        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error("Se ha producido un error al calcular la autocorrelación")

########################## PLOT DECOMPOSOCIÓN ADITIVA ##########################
########################## PLOT DECOMPOSICIÓN ADITIVA ##########################
def plotear_descomposicion_aditiva(df: pd.DataFrame, periodicidad: int) -> None:

    try:
        if periodicidad == 1:
            raise ValueError("No se puede descomponer la serie ya que no tiene periodicidad")
        
        y = df.set_index(df.columns[0]).iloc[:, 0]

        descomposicion = seasonal_decompose(y, model="additive", period=periodicidad)
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        
        axes[0].plot(descomposicion.observed, color="blue")
        axes[0].set_ylabel("Observado")
        
        axes[1].plot(descomposicion.trend, color="orange")
        axes[1].set_ylabel("Tendencia")
        
        axes[2].plot(descomposicion.seasonal, color="green")
        axes[2].set_ylabel("Estacionalidad")
        
        axes[3].plot(descomposicion.resid, color="red")
        axes[3].set_ylabel("Residuos")
        axes[3].axhline(0, color="black", linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    except ValueError as ve:
        st.error(ve)

    except Exception as e:
        st.error("Se ha producido un error ha descomponer la serie" + str(e))

########################## PLOT DECOMPOSOCIÓN MULTIPLICATIVA ##########################
########################## PLOT DECOMPOSICIÓN MULTIPLICATIVA ##########################
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose

def plotear_descomposicion_multiplicativa(df: pd.DataFrame, periodicidad: int) -> None:

    y = df.set_index(df.columns[0]).iloc[:, 0]

    try:
        if y.min() <= 0:
            raise ValueError("El modelo multiplicativo requiere que todos los valores de la serie sean estrictamente positivos.")
        elif periodicidad == 1:
            raise ValueError("No se puede descomponer la serie ya que no tiene periodicidad")

        descomposicion = seasonal_decompose(y, model="multiplicative", period=periodicidad)
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        
        axes[0].plot(descomposicion.observed, color="blue")
        axes[0].set_ylabel("Observado")
        
        axes[1].plot(descomposicion.trend, color="orange")
        axes[1].set_ylabel("Tendencia")
        
        axes[2].plot(descomposicion.seasonal, color="green")
        axes[2].set_ylabel("Estacionalidad")
        
        axes[3].plot(descomposicion.resid, color="red")
        axes[3].set_ylabel("Residuos")
        axes[3].axhline(1, color="black", linestyle="--", alpha=0.5) 
            
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except ValueError as ve:
        st.error(ve)
    except Exception as e:
        st.error("Se ha producido un error al descomponer la serie")