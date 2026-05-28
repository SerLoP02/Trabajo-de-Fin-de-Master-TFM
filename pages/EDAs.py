import streamlit as st
import utils
import pandas as pd


from AI_Agent.Tools.Time_Series_Analysis import time_series_analysis


st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")
st.title("EDA Serie Temporal", text_alignment="center")

serie_original_tab, primera_diff_tab, segunda_diff_tab = st.tabs(["Serie Original", "Primera Diferenciación", "Segunda Diferenciación"])

with serie_original_tab:
    try:
        df_with_null = st.session_state.df.copy()
        col_name = df_with_null.columns[1]
        df_with_null[col_name] = pd.to_numeric(df_with_null[col_name], errors='coerce')
        df_without_null = df_with_null.dropna()

        st.header("Visualización de la Serie Temporal")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_serie(df_with_null)

        st.header("Valores Nulos")
        with st.expander("Pulsa para ver más", True):
            utils.valores_nulos(df_with_null)


        st.header("Estadísticos")
        with st.expander("Pulsa para ver más", True):
            utils.calculo_estadísticos(df_with_null)


        st.header("Análisis Estadístico")
        with st.expander("Pulsa para ver más", True):
            resultados = time_series_analysis(df_without_null.iloc[:, 1])
            utils.mostrar_resultados_serie(resultados)  


        st.header("Función Distribución")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_distribucion(df_without_null)


        st.header("Autoccorelaciones")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_autocorrealciones(df_without_null)


        st.header("Descomposición Aditiva")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_descomposicion_aditiva(df_without_null, st.session_state.periodicidad)


        st.header("Descomposición Multiplicativa")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_descomposicion_multiplicativa(df_without_null, st.session_state.periodicidad)

    except AttributeError as ae:
        st.error("Por favor, sube una serie")


with primera_diff_tab:
    try:
        df_with_null = st.session_state.df.copy()
        col_name = df_with_null.columns[1]
        df_with_null[col_name] = pd.to_numeric(df_with_null[col_name], errors='coerce').diff()
        df_with_null = df_with_null.dropna()
        df_without_null = df_with_null

        st.header("Visualización de la Serie Temporal")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_serie(df_with_null)

        st.header("Estadísticos")
        with st.expander("Pulsa para ver más", True):
            utils.calculo_estadísticos(df_with_null)


        st.header("Análisis Estadístico")
        with st.expander("Pulsa para ver más", True):
            resultados = time_series_analysis(df_without_null.iloc[:, 1])
            utils.mostrar_resultados_serie(resultados)  


        st.header("Función Distribución")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_distribucion(df_without_null)


        st.header("Autoccorelaciones")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_autocorrealciones(df_without_null)


        st.header("Descomposición Aditiva")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_descomposicion_aditiva(df_without_null, st.session_state.periodicidad)


        st.header("Descomposición Multiplicativa")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_descomposicion_multiplicativa(df_without_null, st.session_state.periodicidad)

    except AttributeError as ae:
        st.error("Por favor, sube una serie")



with segunda_diff_tab:
    try:
        df_with_null = st.session_state.df.copy()
        col_name = df_with_null.columns[1]
        df_with_null[col_name] = pd.to_numeric(df_with_null[col_name], errors='coerce').diff().diff()
        df_with_null = df_with_null.dropna()
        df_without_null = df_with_null

        st.header("Visualización de la Serie Temporal")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_serie(df_with_null)

        st.header("Estadísticos")
        with st.expander("Pulsa para ver más", True):
            utils.calculo_estadísticos(df_with_null)


        st.header("Análisis Estadístico")
        with st.expander("Pulsa para ver más", True):
            resultados = time_series_analysis(df_without_null.iloc[:, 1])
            utils.mostrar_resultados_serie(resultados)  


        st.header("Función Distribución")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_distribucion(df_without_null)


        st.header("Autoccorelaciones")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_autocorrealciones(df_without_null)


        st.header("Descomposición Aditiva")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_descomposicion_aditiva(df_without_null, st.session_state.periodicidad)


        st.header("Descomposición Multiplicativa")
        with st.expander("Pulsa para ver más", True):
            utils.plotear_descomposicion_multiplicativa(df_without_null, st.session_state.periodicidad)

    except AttributeError as ae:
        st.error("Por favor, sube una serie")