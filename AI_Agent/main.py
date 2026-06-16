import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
load_dotenv()


# Tools
from AI_Agent.Tools.Time_Series_Analysis import analizar_serie_temporal
from AI_Agent.Tools.Preprocessing import transformar_serie
from AI_Agent.Tools.Series_Plot import plot_serie_temporal
from AI_Agent.Tools.Autocorrelation_Plot import partial_and_autocorrelation_plot
from AI_Agent.Tools.Reset_Serie import reset_transformaciones_serie
from AI_Agent.Tools.Series_Cleansing import imputar_valores_nulos
from AI_Agent.Tools.SARIMAX import entrenar_sarimax
from AI_Agent.Tools.Forecasting import predecir_con_modelo_sarimax
from AI_Agent.Tools.Best_SARIMAX import encontrar_mejor_modelo
from AI_Agent.Tools.AgentState import AgentState


# System Template
from AI_Agent.prompt import SYSTEM_TEMPLATE


# LangChain/LangGraph
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver


tools = [
    analizar_serie_temporal, 
    transformar_serie, 
    plot_serie_temporal,
    partial_and_autocorrelation_plot,
    reset_transformaciones_serie,
    imputar_valores_nulos,
    entrenar_sarimax,
    predecir_con_modelo_sarimax,
    encontrar_mejor_modelo
]

model = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite",
    google_api_key = os.getenv("API_KEY")
).bind_tools(tools)

def model_call(state: AgentState) -> dict:

    # Verificamos si la serie está en el estado actual
    hay_datos = state.get("serie") is not None
    
    # Creamos una instrucción dinámica para el LLM
    if hay_datos:
        instruccion_datos = "**¡ATENCIÓN! El usuario ya te ha adjuntado la serie. Ahora dispones de una serie temporal con la que trabajar.**"
        transformaciones = state["serie_transformed"]["applied_steps"]
        variables_exogenas = list(state["exogenous_vars"].keys())
    else:
        instruccion_datos = "**El usuario aún no ha proporcionado los datos de la serie.**"
        transformaciones = []
        variables_exogenas = []

    system_prompt = SYSTEM_TEMPLATE.format_messages(
    instruccion_datos = instruccion_datos,
    transformaciones = transformaciones,
    variables_exogenas = variables_exogenas
)

    response = model.invoke(system_prompt + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        print("TOOLS LLAMADAS:\n", last_message.tool_calls,"\n\n\n")
        return "continue"
    
graph = StateGraph(AgentState)

graph.set_entry_point("our_agent")
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

memory = InMemorySaver()

app = graph.compile(checkpointer=memory)