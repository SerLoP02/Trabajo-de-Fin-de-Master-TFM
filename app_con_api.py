import streamlit as st
import base64
import pandas as pd
import requests
import os


from utils import seleccionar_periodicidad
from File_Reader.main import identificar_formato_serie


# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Agente de Series Temporales", page_icon="📈", layout="wide")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- ESTADO DE LA SESIÓN DE STREAMLIT ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "usuario_1"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Sube un dataset en la barra lateral y pregúntame lo que quieras sobre tu serie temporal."}]

if "id_archivo" not in st.session_state:
    st.session_state.id_archivo = ""
    st.session_state.archivo = None
    st.session_state.df = None

if "iniatilize_state" not in st.session_state:
    st.session_state.iniatilize_state = True
    payload = {
        "serie": None,
        "serie_transformed": None,
        "periodicidad": None,
        "formato_fecha": None,
        "exogenous_vars": {}
    }
    try:
        response = requests.patch(f"{API_URL}/update/{st.session_state.thread_id}", json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al conectar con la API para inicializar el estado del agente: {e}")

if "periodicidad" not in st.session_state:
    st.session_state.periodicidad = None

# --- BARRA LATERAL (Carga de Datos) ---
with st.sidebar:
    st.header("📂 Carga de Datos")
    
    if (uploaded_file := st.file_uploader("Sube tu archivo CSV", type=["csv"])) is not None: 

        id_archivo = uploaded_file.file_id
        periodicidad = seleccionar_periodicidad()
            
        # Si cambia el archivo o cambia la periodicidad, asumo que el usuario quiere realizar un análisis nuevo
        if st.session_state.id_archivo != id_archivo or st.session_state.periodicidad != periodicidad:
            try:
                try:
                    formato = identificar_formato_serie(uploaded_file)
                except Exception:
                    st.warning("⚠️ Ha habido un problema al identificar el delimitador, el separador decimal o el formato de fechas. Se han asignado los valores estándar americanos.")
                    formato = {"delimitador": ",", "separador_decimal": ".", "formato_fecha": "%m-%d-%Y"}
                st.write("Formato detectado: ", formato)

                st.session_state.thread_id += "1"
                df = pd.read_csv(uploaded_file, decimal=formato["separador_decimal"], delimiter=formato["delimitador"], parse_dates=[0], date_format=formato["formato_fecha"])
                
                if df.isna().any().any():
                    from numpy import nan
                    df = df.replace({nan: None}) # JSON no entiende de np.nan
                    st.warning("⚠️ La serie temporal que has enviado contiene valores nulos.")

                time_values_vars = df.columns.tolist()[:2]

                exogenous_vars = df.columns.tolist()[2:]
                exogenous_vars = df.loc[:, exogenous_vars].to_dict("list")

                datos_serie = {
                    "values": df.iloc[:, 1].tolist(), # ¡OJO! Asumo valores en la 2da columna
                    "index": df.iloc[:, 0].astype(str).tolist(), # ¡OJO! Asumo índice en la 1ra columna
                    "name_cols": time_values_vars,
                    "applied_steps": [],
                }
                
                payload = {
                    "serie": datos_serie,
                    "serie_transformed": datos_serie,
                    "periodicidad": periodicidad,
                    "exogenous_vars": exogenous_vars,
                    "formato_fecha": formato["formato_fecha"]
                }
                response = requests.patch(f"{API_URL}/update/{st.session_state.thread_id}", json=payload)
                response.raise_for_status()
                
                st.success("¡Datos cargados en el agente correctamente!")

                st.session_state.id_archivo = id_archivo
                st.session_state.archivo = uploaded_file
                st.session_state.df = df
                st.session_state.periodicidad = periodicidad

            except Exception as e:
                st.error(f"Error al procesar el archivo o contactar con la API: {e}")               

# --- INTERFAZ DE CHAT (muestra de mensajes y artifactos continua) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "artifacts" in msg and msg["artifacts"]:
            for artifact in msg["artifacts"]:
                if artifact.get("summary"):
                    with st.expander("Ver detalles estadísticos del modelo entrenado"):
                        st.text(artifact["summary"])
                if artifact.get("image"):
                    image_bytes = base64.b64decode(artifact["image"])
                    st.image(image_bytes)

if prompt := st.chat_input("Escribe tu pregunta o pide un gráfico..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando y consultando herramientas...", show_time=True):
            
            payload = {"user_input": prompt}
           
            try:
                response = requests.post(f"{API_URL}/invoke/{st.session_state.thread_id}", json=payload)
                response.raise_for_status()
                data = response.json()
                mensajes_nuevos = data.get("user_output", [])

                # Imprimimos por pantalla los mensajes (DEBBUGING)
                for message in mensajes_nuevos: 
                    tipo_mensaje = message.get("type").upper()
                    print(f"============{tipo_mensaje}============")
                    contenido = message.get("content", "")
                    print(contenido, "\n", "type -> ", type(contenido), "\n")
                    if message.get("tool_calls"):
                        print(message.get("tool_calls"), "\n")
            except requests.exceptions.RequestException as e:
                st.error(f"Error al comunicarse con la API: {e}")
                mensajes_nuevos = []
                st.stop()

            from utils import extraer_texto_limpio_con_api
            respuesta_final = extraer_texto_limpio_con_api(mensajes_nuevos[-1])
            st.markdown(respuesta_final)
            
            # Comprobamos si en los mensajes generados hay artifactos
            collected_artifacts = []
            for msg in mensajes_nuevos:
                if msg.get("type") == "tool" and msg.get("artifact"):
                    collected_artifacts.append(msg.get("artifact"))
            
            for artifact in collected_artifacts:
                if artifact.get("summary"):
                    with st.expander("Ver detalles estadísticos del modelo", expanded=True):
                        try:
                            st.text(artifact["summary"])
                        except Exception as e:
                            st.error("No se pudo cargar el resumen estadístico")
                if artifact.get("image"):
                    image_bytes = base64.b64decode(artifact["image"])
                    st.image(image_bytes)

    # Actualizamos los mensajes y artifactos de la sesión para que se muestren continuamente    
    st.session_state.messages.append({
        "role": "assistant",
        "content": respuesta_final,
        "artifacts": collected_artifacts
    })