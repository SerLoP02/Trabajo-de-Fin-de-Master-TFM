from fastapi import FastAPI, status
from pydantic import BaseModel, Field


from langchain_core.messages import HumanMessage


from AI_Agent.main import app


class UserRequest(BaseModel):
    user_input: str

class UserResponse(BaseModel):
    user_output: list[dict]

class UpdateStateRequest(BaseModel):
    serie: dict | None = None
    serie_transformed: dict | None = None
    periodicidad: int | None = Field(default=None, ge=1)
    formato_fecha : str | None = None
    exogenous_vars : dict

api = FastAPI()

@api.get("/")
def welcome():
    return {"message": "Bienvenido"}

@api.post("/invoke/{thread_id}", response_model=UserResponse)
def invoke(user_request: UserRequest, thread_id: str) -> dict:
    """
    Endpoint para responder al input del usuario
    """

    config = {"configurable": {"thread_id": thread_id}}

    estado_previo = app.get_state(config).values
    num_mensajes_previos = len(estado_previo.get("messages", []))

    new_message = {"messages": [HumanMessage(content=user_request.user_input)]}
    respuesta_estado = app.invoke(new_message, config).get("messages")
    mensajes_nuevos = respuesta_estado[num_mensajes_previos:]
    mensajes_nuevos = [message.model_dump() for message in mensajes_nuevos]
    return {"user_output": mensajes_nuevos}

@api.patch("/update/{thread_id}", status_code=status.HTTP_201_CREATED)
def update_state(new_state: UpdateStateRequest, thread_id: str) -> None:
    """
    Endpoint para inicializar el estado del agente, ya sea cuando es la primera vez que se ejecuta
    o cuando el usuario sube una serie
    """
    config = {"configurable": {"thread_id": thread_id}}

    values = {**new_state.model_dump(), "messages": [], "sarimax_model": None, }
    app.update_state(config, values)
    return new_state.model_dump()