import io
import base64
import pandas as pd
from typing import Annotated
import matplotlib.pyplot as plt


from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from Errores.main import NullValuesError, NoSerieError

@tool(response_format="content_and_artifact")
def partial_and_autocorrelation_plot(
    serie_transformed: Annotated[dict | None, InjectedState("serie_transformed")]
):
    """
    Genera los gráficos de autocorrelación (ACF) y autocorrelación parcial (PACF)
    a partir de la serie temporal almacenada en 'serie_transformed'.

    Debes usar esta herramienta cuando el usuario pida graficar o ver cualquier tipo de autocorrelación, 
    ya sea la parcial (PATF) o la total (ATF)

    Qué hace:
        - Recupera la serie almacenada en 'serie_transformed'.
        - Genera los gráficos ACF y PACF con un número fijo de rezagos.
        - Envía la imagen al sistema para que se muestre automáticamente al usuario.
    """

    try:
        if not serie_transformed:
            raise NoSerieError("Error: No hay datos de serie temporal para graficar.")
        
        y_t = pd.Series(serie_transformed["values"], index=pd.to_datetime(serie_transformed["index"]))

        if y_t.isnull().any().any():
            raise NullValuesError("Error: Hay valores nulos en la serie y no se puede hallar la autocorrelación.")

        num_lags = 15

        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        plot_acf(y_t, ax=ax[0], lags=num_lags, title="Autocorrelación")
        plot_pacf(y_t, ax=ax[1], lags=num_lags, method="ols", title="Autocorrelación parcial")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()

        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        content_for_llm = "Éxito: El gráfico ha sido generado"
        return content_for_llm, {"image": image_base64}
    
    except NoSerieError as nse:
        return str(nse), {}
    
    except NullValuesError as nve:
        return str(nve), {}
    
    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle.", {}