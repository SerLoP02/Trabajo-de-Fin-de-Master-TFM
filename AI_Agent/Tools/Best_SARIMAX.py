

from typing import Annotated
import pandas as pd
import matplotlib.pyplot as plt
import io
import pickle
import base64


from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX


from Errores.main import NoSerieError, NullValuesError

@tool
def encontrar_mejor_modelo(
    serie_transformed: Annotated[dict | None, InjectedState("serie_transformed")],
    serie: Annotated[dict | None, InjectedState("serie")],
    exog: Annotated[dict, InjectedState("exogenous_vars")],
    periodicidad: Annotated[int | None, InjectedState("periodicidad")],
    tool_call_id: Annotated[str, InjectedToolCallId],
    exogenous_variables: list[str] | None = None
):
    """
    Hace una búsqueda del mejor modelo SARIMAX.
    
    Úsala cuando:
        - El usuario solicite entrenar o predecir un modelo SARIMAX pero no especifique cuál

    Qué hace:
        - Encuentra el mejor modelo SARIMAX y le guarda en el estado 'sarimax_model'

    Argumentos
        - exogenous_variables: Lista con los nombres de las variables exógenas a usar. Si el usuario no especifica ninguna variable exógena, no uses ninguna variable exógena (el valor por defecto es None, es decir, ninguna variable exógena)

    Retorna:
        - Una imagen con los plots de diagnóstico y el sumario estadístico.
    """

    try:
        if not serie_transformed or not serie:
            raise NoSerieError("Error: No hay datos de serie temporal con la que trabajar")
        
        applied_steps = serie_transformed["applied_steps"]

        y_t = pd.Series(data=serie_transformed["values"])

        # Comprobamos si la serie tiene valores nulos y el usuario ya los ha quitado
        has_null = y_t.isnull().any()
        metodos_imputacion = ["Imputamos valores nulos mediante interpolacion_lineal",
                              "Imputamos valores nulos mediante spline_cubico"]
        has_imputation_step = any(step in applied_steps for step in metodos_imputacion)

        if has_null and has_imputation_step:
            metodo = next((step for step in applied_steps if step in metodos_imputacion), None)
            metodo_pandas = "linear" if "interpolacion_lineal" in metodo else "cubic"
            y_t = y_t.interpolate(method=metodo_pandas)
        elif has_null:
            raise NullValuesError("Error: La serie contiene valores nulos")

        train_len = int(len(y_t) * 0.8)
        y_train = y_t.iloc[:train_len]

        if exogenous_variables and exog:
            model_exog = pd.DataFrame(exog)[exogenous_variables].iloc[:train_len, :]
        else:
            model_exog = None

        best_model = auto_arima(
            y = y_train,
            X = model_exog,
            m = periodicidad
        )

        best_model = SARIMAX(
            endog = y_train,
            exog = model_exog,
            order = best_model.order,
            seasonal_order = best_model.seasonal_order
        )
        results = best_model.fit(disp=False)

        # SUMMARY
        summary = results.summary()
        summary_str = summary.as_text()

        # RESIDUAL PLOTS
        results.plot_diagnostics(figsize=(10,4))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        mensaje_exito = "Se ha encontrado el mejor modelo de manera exitosa"

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
    
    except NullValuesError as nve:
        return str(nve)
    
    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle."