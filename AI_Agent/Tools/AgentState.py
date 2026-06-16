from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence, TypedDict


class AgentState(TypedDict):
    
    # MEMORIA
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # SERIE TEMPORAL
    serie: dict | None
    serie_transformed: dict | None
    periodicidad: int | None
    formato_fecha: str | None

    # SARIMAX
    sarimax_model: bytes | None
    exogenous_vars: dict