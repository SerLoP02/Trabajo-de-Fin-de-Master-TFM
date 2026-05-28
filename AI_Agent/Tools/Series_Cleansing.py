from abc import ABC, abstractmethod
import pandas as pd
from typing import Annotated, Literal


from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command


class BaseImputer(ABC):

    @abstractmethod
    def impute(self, series: pd.Series | pd.DataFrame) -> pd.DataFrame:
        pass

class LinearInterpolationImputer(BaseImputer):
    
    def impute(self, series):
        return series.interpolate(method="linear")

class SplineImputer(BaseImputer):

    def impute(self, series):
        return series.interpolate(method="spline", order=3)
    
Imputers = {
    "interpolacion_lineal": LinearInterpolationImputer(),
    "spline_cubico": SplineImputer()
}

class TimeSeriesImputer:
    def __init__(self, method: str):
        if method not in Imputers:
            raise ValueError(f"Método no soportado. Actualmente solo se pueden usar los métodos 'spline_cubico' y 'interpolacion_lineal'")
        self.imputer = Imputers[method]

    def __call__(self, series: pd.Series) -> pd.Series:
        return self.imputer.impute(series)

@tool
def imputar_valores_nulos(
    serie_transformed: Annotated[dict | None, InjectedState("serie_transformed")], 
    exogenous_vars: Annotated[dict, InjectedState("exogenous_vars")],
    tool_call_id: Annotated[str, InjectedToolCallId],
    method: Literal["interpolacion_lineal", "spline_cubico"]
):
    """Herramienta para el tratamiento de valores faltantes

    Parámetros:
        method (str): Método a usar:
            - 'interpolacion_lineal': Realiza una interpolación lineal. Si el usuario dice de hacer una interpolación pero no especifica cuál, asume que se refiere a una interpolación lineal
            - 'spline_cubico': Realiza un spline cúbico
            
    Salida:
        Actualiza 'serie_transformed' con los mismos valores de la serie original salvo que ahora los valores nulos se han imputado mediante el 'method' elegido
        
    *NOTAS*:
        - Si el usuario no especifica ningún 'method' o elige uno no disponible, debes indicarle las opciones disponibles antes de usar la herramienta
    """
    

    try:
        y_t = pd.Series(
            serie_transformed["values"]
        )

        imputer = TimeSeriesImputer(method)

        y_t = imputer(y_t)

        mensaje_exito = f"Se han imputado los valores nulos de la serie con el método {method} de manera exitosa"
        step = f"Imputamos valores nulos mediante {method}"

        serie_transformed_dict = {
            "values": y_t.values.tolist(),
            "index": serie_transformed["index"],
            "name_cols": serie_transformed["name_cols"],
            "applied_steps": serie_transformed["applied_steps"] + [step]
        }

        update_dict = {
            "serie_transformed": serie_transformed_dict,
            "messages": [ToolMessage(content=mensaje_exito, tool_call_id=tool_call_id)]
        }

        if exogenous_vars:
            
            df_exog = pd.DataFrame(exogenous_vars)            
            df_imputed = df_exog.apply(imputer) 

            update_dict["exogenous_vars"] = df_imputed.to_dict("list")

        return Command(
            update = update_dict
        )
    
    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle."
