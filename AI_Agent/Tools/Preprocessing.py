import pandas as pd
import numpy as np
from typing import Annotated


from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState


from Errores.main import NoSerieError, NullValuesError

def preprocess_time_series(
    y_t: pd.Series,
    apply_log: bool, 
    diff_order: int
):
    
    processed_series = y_t.copy()
    applied_steps = []

    # Transformación Logarítmica (Estabilizar varianza)
    if apply_log:
        # Asegurarnos de que no hay valores <= 0 antes de aplicar log
        if (processed_series > 0).all():
            processed_series = np.log(processed_series)
            applied_steps.append("Aplicamos logaritmo.")
        else:
            raise ValueError("Error: La serie tiene valores nulos o negativos y no se puede aplicar el logaritmo.")

    # Diferenciación (Estabilizar la media)
    if diff_order > 0:
        for _ in range(diff_order):
            processed_series = processed_series.diff()
            applied_steps.append(f"Aplicamos diferenciación de orden 1")
        processed_series = processed_series.dropna()
    return (applied_steps, processed_series)

@tool
def transformar_serie(
    state: Annotated[dict, InjectedState], 
    tool_call_id: Annotated[str, InjectedToolCallId],
    diff_order: int,
    apply_log: bool = False
    ):
    """
    Transforma la serie temporal almacenada en 'serie_transformed'

    Debes usarla cuando el usuario pida:
        - aplicar logaritmo,
        - diferenciar la serie,
        - hacer la serie estacionaria (en este caso, usa también la tool 'analizar_serie_temporal' para determinar estacionariedad),
        - eliminar tendencia o estabilizar la varianza,
        - etc.

    Qué hace:
        - Si 'apply_log' es True, aplica logaritmo siempre que todos los valores sean positivos.
        - Aplica una diferenciación de orden 'diff_order' (2 como mucho).
        - Actualiza 'serie_transformed' con la nueva serie procesada.

    Parámetros:
        apply_log (bool):
            True -> aplicar logaritmo si es posible.
            False -> no aplicar logaritmo. Si el usuario no especifica, el valor por defecto será False.
        diff_order (int):
            Número de veces que se diferencia la serie (2 como mucho). Si el usuario hace alusión a una diferenciación y no especifica el orden, por ejemplo dice 'diferencia la serie', se asume 'diff_order' = 1

    Salida:
        Actualiza el estado 'serie_transformed'.
    """

    try:
        if not state["serie_transformed"]:
            raise NoSerieError("Error: No hay datos de serie temporal para hacer la transformación.")
        
        y_t = pd.Series(
            state["serie_transformed"]["values"],
            index=state["serie_transformed"]["index"]
        )

        if y_t.isnull().any():
            raise NullValuesError("Error: Hay valores nulos en la serie y no se pueden aplicar las transformaciones.")

        applied_steps, serie_transformada = preprocess_time_series(y_t, apply_log, diff_order)

        mensaje_exito = "La serie ha sido transformada de manera exitosa"

        serie_transformada_dict = {
            "values": serie_transformada.values.tolist(),
            "index": serie_transformada.index.astype(str).tolist(),
            "name_cols": state["serie_transformed"]["name_cols"],
            "applied_steps": state["serie_transformed"]["applied_steps"] + applied_steps
        }


        return Command(
            update={
                "serie_transformed": serie_transformada_dict,
                "messages": [ToolMessage(content=mensaje_exito, tool_call_id=tool_call_id)]
            }
        )
    
    except NoSerieError as nse:
        return str(nse)
    
    except NullValuesError as nve:
        return str(nve)
    
    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle."