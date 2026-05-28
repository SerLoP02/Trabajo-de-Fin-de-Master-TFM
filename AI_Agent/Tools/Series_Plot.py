import matplotlib.pyplot as plt
import io
import base64
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
import pandas as pd
from Errores.main import NoSerieError

@tool(response_format="content_and_artifact")
def plot_serie_temporal( 
    serie_transformed: Annotated[dict | None, InjectedState("serie_transformed")]
):
    """
    Genera un gráfico de la serie temporal y lo muestra al usuario.

    Debes usar esta herramienta cuando el usuario pida:
        - visualizar la serie temporal,
        - dibujar un gráfico,
        - ver la evolución de los datos en el tiempo,
        - obtener una representación visual de la serie previamente procesada,
        - etc.

    Qué hace:
        - Recupera la serie almacenada en 'serie_transformed'
        - Genera una imagen de la serie y la envía al sistema para que se muestre automáticamente al usuario
    """
    try:
        if not serie_transformed:
            raise NoSerieError("Error: No hay datos de serie temporal para graficar.")
        
        y_t = pd.Series(serie_transformed["values"], index=pd.to_datetime(serie_transformed["index"]))
        name_cols = serie_transformed["name_cols"]

        plt.figure(figsize=(10, 4))
        plt.plot(pd.to_datetime(y_t.index), y_t, linestyle='-', color='#1f77b4')
        plt.title("Visualización de la Serie Temporal")
        plt.xlabel(name_cols[0])
        plt.ylabel(name_cols[1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        content_for_llm = (
        "Éxito: El gráfico ha sido generado"
    )
        
        return content_for_llm, {"image": image_base64}
    
    except NoSerieError as nse:
        return str(nse), {}
    
    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle.", {}