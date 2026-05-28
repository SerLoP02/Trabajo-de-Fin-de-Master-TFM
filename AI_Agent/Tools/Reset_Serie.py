from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from AI_Agent.Tools.AgentState import AgentState
from Errores.main import NoSerieError

from typing import Annotated

@tool
def reset_transformaciones_serie(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Esta herramienta resetea cualquier transformación que se haya hecho previamente a la serie: retorna la serie almacenada en 'serie_transformed'
    a su estado original, 'serie'"""

    try: 
        if not state["serie"]:
            raise NoSerieError("Error: No hay datos de serie temporal.")

        mensaje_exito = "La serie ha sido reseteada exitosamente. Cualquier cambio hecho previamente se ha deshecho."

        return Command(
            update = {
                "serie_transformed": state["serie"],
                "messages": [ToolMessage(content=mensaje_exito, tool_call_id=tool_call_id)]
            }
        )
    
    except NoSerieError as nse:
        return str(nse)

    except Exception as e:
        return f"Se ha producido el siguiente error: {e}. Intenta explicar el error al analista para que pueda corregirle."