import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import statsmodels.api as sm
from scipy.stats import shapiro, ttest_1samp
from AI_Agent.Tools.AgentState import AgentState
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated, Dict, Union

from Errores.main import NoSerieError, NullValuesError

def Newey_West_OLS(y_t: pd.Series) -> float:
    """
    Test riguroso de H0: media = 0 en serie temporal,
    usando OLS + errores estándar HAC (Newey–West).
    """

    # Creamos un vector de unos (una constante)
    X = np.ones((len(y_t), 1))

    # Ajustamos un modelo OLS (Ordinary Least Squares) de y_t sobre la constante
    model = sm.OLS(y_t, X)

    # Ajustamos el modelo usando errores estándar HAC (Newey-West)
    # maxlags define la ventana de rezagos. Si se omite o se pone None, 
    # se suele usar una heurística, pero aquí le damos un valor fijo razonable (ej. 10) 
    # o puedes calcularlo dinámicamente como int(4 * (len(y_t)/100)**(2/9))
    lags = int(4 * (len(y_t) / 100)**(2/9))
    robust = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})

    return float(robust.pvalues.iloc[0])


def time_series_analysis(y_t: pd.Series) -> dict:

    # ============================================================
    # 1. ESTACIONARIEDAD (ADF)
    # H0: la serie NO es estacionaria
    adf_pvalue = float(adfuller(y_t)[1])
    is_stationary = adf_pvalue < 0.05
    # ============================================================    

    # ============================================================
    # 2. TIENE MEDIA CERO (T-test + Newey-West)
    t_pvalue = float(ttest_1samp(y_t, popmean=0.0).pvalue)
    nw_pvalue = Newey_West_OLS(y_t)

    mean_is_zero = nw_pvalue > 0.05    
    # ============================================================

    # ============================================================
    # 3. AUTOCORRELACIÓN LINEAL (Ljung–Box)
    # H0: no hay autocorrelación lineal
    lb_pvalue = float(acorr_ljungbox(y_t, lags=[25])["lb_pvalue"].iloc[0])
    no_linear_autocorr = lb_pvalue > 0.05
    # ============================================================ 

    # ============================================================
    # 3. AUTOCORRELACIÓN ORDEN SUPERIOR (Ljung–Box sobre y^n)
    # H0: no hay autocorrelación lineal
    for exp in range(2, 4):
        lb_pvalue_orden_superior = float(acorr_ljungbox(y_t**exp, lags=[25])["lb_pvalue"].iloc[0])
        no_nonlinear_autocorr = lb_pvalue_orden_superior > 0.05
        if not no_nonlinear_autocorr:
            break
    # ============================================================ 

    # Spearman y Kendall tests

    # ============================================================
    # 5. NORMALIDAD (Shapiro)
    # H0: distribución normal
    shapiro_pvalue = float(shapiro(y_t)[1])
    is_normal = shapiro_pvalue > 0.05
    # ============================================================

    is_WN = is_stationary and mean_is_zero and no_linear_autocorr

    is_not_SWN = is_WN and not no_nonlinear_autocorr


    if is_WN:
        if is_not_SWN:
            info_SWN = f"La serie no es Ruido Blanco Estricto. Presenta autocorrelación de orden {exp}. El Ljung-Box pvalor de y^{exp} es {lb_pvalue_orden_superior}"
            info_GWN = f"La serie no es Ruido Blanco Gaussiano porque la serie no es completamente independiente. Presenta dependecia de orden y^{exp}"
        else:
            info_SWN = f"La serie no presenta autocorrelación hasta y^{exp} (ljung-box pvalor: {lb_pvalue_orden_superior}). Podría ser Ruido Blanco Estricto, pero es necesario hacer un análisis más detallado"
            if is_normal:
                info_GWN = f"La serie no presenta autocorrelación hasta y^{exp} (ljung-box pvalor: {lb_pvalue_orden_superior}) y además presenta distribución normal (el pvalor para el Shapiro Test es {shapiro_pvalue}). Podría ser Ruido Blanco Gaussiano, pero habría que hacer un análisis más detallado"
            else:
                info_GWN = f"La serie no puede ser Ruido Blanco Gaussiano porque no tiene distribución normal. El pvalor para el Shapiro Test es {shapiro_pvalue}."
    else:
        info_SWN = f"La serie no puede ser Ruido Blanco Estricto porque no es Ruido Blanco"
        info_GWN = f"La serie no puede ser Ruido Blanco Gaussiano porque no es Ruido Blanco"

    result = {
        "Estacionaridad": {
            "Es estacionaria": is_stationary,
            "ADF pvalor": adf_pvalue
        },
        "Ruido Blanco": {
            "Es Ruido Blanco": is_WN,
            "Media zero": {
                "Tiene media cero": mean_is_zero,
                "Newey-West-OLS pvalor": nw_pvalue
            },
            "No Autocorrelacion Lineal": {
                "No Autocorrelacion Lineal": no_linear_autocorr,
                "Ljung-Box pvalor": lb_pvalue
            }                    
        },
        "Ruido Blanco Estricto": {
            "Informacion": info_SWN
        },
        "Ruido Blanco Gaussiano": {
            "Información": info_GWN
        }
    }

    return result


@tool
def analizar_serie_temporal(
    serie_transformed: Annotated[dict | None, InjectedState("serie_transformed")]
):
    """ Analiza la serie temporal almacenada en 'serie_transformed' y devuelve
    un diagnóstico estadístico completo sobre sus propiedades fundamentales
    
    Este análisis incluye (se incluyen también los pvalores):
        1. **Estacionariedad (ADF Test)**  
        2. **Media igual a cero (Newey–West Test)**  
        3. **Autocorrelación lineal (Ljung–Box Test)**  
        4. **Autocorrelación no lineal (Ljung–Box Test sobre potencias y^2 e y^3)**  
        5. **Normalidad (Shapiro–Wilk)**  
        6. **Clasificación del proceso**  
            Con base en los resultados anteriores, determina si la serie puede considerarse:
            - Estacionaria
            - Ruido Blanco (WN)
            - Ruido Blanco Estricto (SWN)
            - Ruido Blanco Gaussiano (GWN)
    """

    try:  
        if not serie_transformed:
            raise NoSerieError("Error: No hay datos de serie temporal para graficar.")
        
        y_t = pd.Series(
            serie_transformed["values"],
            index = pd.to_datetime(serie_transformed["index"])
        )

        if y_t.isnull().any():
            raise NullValuesError("Error: hay valores nulos en la serie y no se puede hacer el análisis.")

        resultado = time_series_analysis(y_t)
        
        return f"Aquí tienes los resultados del análisis:\n{resultado}"
    
    except NoSerieError as nse:
        return str(nse)
    
    except NullValuesError as nve:
        return str(nve)
    
    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle."