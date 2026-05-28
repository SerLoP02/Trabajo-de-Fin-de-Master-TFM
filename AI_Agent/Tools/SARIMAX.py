import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas import DataFrame
import io
import base64
import pickle
from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState


from Errores.main import NoSerieError

@tool
def entrenar_sarimax(
    serie_transformed: Annotated[dict | None, InjectedState("serie_transformed")],
    exog: Annotated[dict, InjectedState("exogenous_vars")],
    periodicidad: Annotated[int | None, InjectedState("periodicidad")],
    tool_call_id: Annotated[str, InjectedToolCallId],
    order: list[int],
    seasonal_order: list[int],
    exogenous_variables: list[str] | None = None
):
    """
    Entrena un modelo estadístico SARIMAX para el análisis y pronóstico de series temporales.
    
    Úsala cuando el usuario solicite entrenar, predecir, etc. modelos ARIMA, SARIMA, AR, MA, ARMA, S-AR, S-MA, S-ARMA con o sin variables exógenas.

    Qué hace:
        - Usa el estado 'serie_transformed' para entrenar el modelo especificado usando el primer 80% de los datos
        - Actualiza el modelo entrenado en el estado 'sarimax_model'
        - Devuelve una imagen con los plots de diagnóstico y el sumario estadístico

    Argumentos:
    - order: Una lista [p, d, q] para la parte no estacional:
        * p: Orden de la parte autorregresiva (AR).
        * d: Grado de diferenciación para hacer la serie estacionaria (I).
        * q: Orden de la parte de media móvil (MA).
    - seasonal_order: Una lista [P, D, Q] para la parte estacional:
        * P: Orden autorregresivo estacional (S-AR).
        * D: Grado de diferenciación estacional.
        * Q: Orden de media móvil estacional (S-MA).
    - exogenous_variables: Lista con los nombres de las variables exógenas a usar. Si el usuario no especifica ninguna variable exógena, no uses ninguna variable exógena (el valor por defecto es None, es decir, ninguna variable exógena)

    Nota de implementación:
        - seasonal order es una lista de tres elementos. El paramétro S ya ha sido proporcionado por el analista y no es necesario que lo vuelvas a pedir
    """
    
    try:
        if not serie_transformed:
            raise NoSerieError("Error: No hay datos de serie temporal con la que realizar el entrenamiento.")
        
        if periodicidad == 1:
            if seasonal_order != [0, 0, 0]:
                raise ValueError("Error: El analista ha establecido que la serie no tiene periodicidad, pero ha eligido estacionalidad en el modelo. Esto es incongruente, házselo saber")
            else:
                seasonal_order = [0, 0, 0, 0]
        else:
            seasonal_order.append(periodicidad)

        train_len = int(len(serie_transformed["values"]) * 0.8)
        y_train = serie_transformed["values"][:train_len]

        model_exog = None 
        
        if exogenous_variables:
            model_exog = DataFrame(exog).loc[:, exogenous_variables].iloc[:train_len, :]

        model = SARIMAX(
            endog = y_train,
            exog = model_exog,
            order = order,
            seasonal_order=seasonal_order,
            enforce_invertibility = False,
            enforce_stationarity = False   
        )
        results = model.fit(disp=False)

        # SUMMARY
        summary = results.summary()
        summary_str = summary.as_text()

        # RESIDUAL PLOTS
        results.plot_diagnostics(figsize=(10, 4))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        mensaje_exito = "Modelo entrenado de manera exitosa"

        results_bytes = pickle.dumps(results)
        command = Command(
            update = {
                "sarimax_model": results_bytes, 
                "messages": [ToolMessage(content=mensaje_exito, tool_call_id=tool_call_id, artifact={"image": image_base64, "summary": summary_str})]
            }
        )

        return command
    
    except NoSerieError as nse:
        return str(nse)
    
    except ValueError as ve:
        return str(ve)

    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle."