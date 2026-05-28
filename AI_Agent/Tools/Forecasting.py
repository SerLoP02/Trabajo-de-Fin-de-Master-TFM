import pickle
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Annotated


from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

def rmse(y_true, y_test) -> float:
    rmse = np.sqrt(np.mean(np.square(y_true - y_test)))
    return rmse

@tool(response_format="content_and_artifact")
def predecir_con_modelo_sarimax(
    sarimax_model: Annotated[bytes | None, InjectedState("sarimax_model")],
    serie_transformed: Annotated[dict | None, InjectedState("serie_transformed")],
    exog: Annotated[dict | None, InjectedState("exogenous_vars")],
    exogenous_variables: list[str] | None = None
):
    """Herramienta para predecir, evaluar y visualizar el rendimiento de un modelo SARIMAX previamente entrenado
    
    Qué hace:
        - Recuperda el último modelo entrenado almacenado en 'sarimax_model'
        - Predice sobre el último 20% de los datos de la serie 'serie_transformed'

    Argumentos
        - exogenous_variables: Lista con los nombres de las variables exógenas usadas durante el entrenamiento del modelo (el usado en la herramienta 'encontrar_mejor_modelo' o 'entrenar_sarimax').

    Úsala cuando:
        - El usuario pida realizar o ejecutar la predicción con un modelo SARIMAX,
        - El usuario quiera ver la gráfica comparativa (Real vs Predicción),
        - etc.
        
    Retorna:
        Una imagen con la comparativa de la predicción vs el resultado real y el RMSE"""

    try:
        if not sarimax_model:
            raise KeyError("No has entrenado ningún modelo. Tienes que entrenar el modelo antes de usar esta herramienta")
        
        train_size = int(0.8 * len(serie_transformed["values"]))
        test_serie = serie_transformed["values"][train_size:]

        if exogenous_variables:
            model_exog = pd.DataFrame(exog)[exogenous_variables].iloc[train_size:, :]
        else:
            model_exog = None

        sarimax_model = pickle.loads(sarimax_model)

        pred_serie = sarimax_model.forecast(steps=len(test_serie), exog=model_exog)

        test_serie = np.array(test_serie)

        bondad = 100 * (1 - rmse(test_serie, pred_serie) / test_serie.mean())
        bondad = round(bondad, 3)

        plt.figure(figsize=(10, 4))
        plt.plot(test_serie, label="Real")
        plt.plot(np.array(pred_serie).squeeze(), c="r", label="Predicción")
        plt.title(f"Bondad del ajuste: {bondad}%")
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()

        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        content_for_llm = f"Predicción exitosamente realizada. Bondad del ajuste: {bondad}%"

        return content_for_llm, {"image": image_base64}
    
    except KeyError as ke:
        return str(ke), {}
    
    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle.", {}