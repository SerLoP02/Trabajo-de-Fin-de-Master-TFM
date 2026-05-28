import csv
import io
import re
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pandas.tseries.api import guess_datetime_format

def detectar_separador_decimal(lineas_texto: list, delimitador: str) -> str:
    valores_serie = [row.split(delimitador)[1] for row in lineas_texto] # ¡OJO! Asumo valores de la serie o exógenas en la segunda columna. Si la fecha está aquí, fallará

    # Si el delimitador es ",", el separador decimal necesariamente es "."
    if delimitador == ',':
        return '.'
        
    # Si el delimitador es ";", necesitamos crear una lógica para determinar si el separador decimal es "," ó "."
    patron_punto = re.compile(r'\d+\.\d+')
    patron_coma = re.compile(r'\d+,\d+')
    
    conteo_puntos = 0
    conteo_comas = 0

    valores_serie = [row.split(delimitador)[1] for row in lineas_texto] # ¡OJO! Asumo valores de la serie o exógenas en la segunda columna. Si la fecha está aquí, fallará
    for valor_serie in valores_serie:
        if patron_punto.search(valor_serie):
            conteo_puntos += 1
        if patron_coma.search(valor_serie):
            conteo_comas += 1
            
    if conteo_comas > conteo_puntos:
        return ','
        
    # Por defecto devolvemos el punto
    return '.'



def identificar_formato_serie(uploaded_file: UploadedFile) -> dict:
    
    uploaded_file.seek(0)
    # Leemos la muestra
    lineas_csv = [uploaded_file.readline().decode('utf-8').strip() for _ in range(14) if uploaded_file.readline()]
    archivo_csv_str = "\n".join(lineas_csv)
    uploaded_file.seek(0)

    # ---  DETECTAR DELIMITDOR ---
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(archivo_csv_str)
        delimitador = dialect.delimiter
    except csv.Error:
        delimitador = ","

    # --- DETECTAR SEPARADOR DECIMAL ---
    separador_decimal = detectar_separador_decimal(lineas_csv, delimitador)
    
    # --- DETECTAR FORMATO FECHA ---
    buffer_str = io.StringIO(archivo_csv_str)
    filas_csv = csv.reader(buffer_str, delimiter=delimitador)

    formato_fecha_detectado = None
    
    for fila in filas_csv:
        if len(fila) > 0:
            posible_fecha = fila[0]  # ¡OJO! Asumo que la fecha está en la columna 0
            
            # Intentamos adivinar. Si es texto (ej. "Fecha"), devolverá None
            formato = guess_datetime_format(posible_fecha)
            
            if formato is not None:
                formato_fecha_detectado = formato
                break  # Encontramos el formato, dejamos de buscar

    return {
        "delimitador": delimitador,
        "separador_decimal": separador_decimal,
        "formato_fecha": formato_fecha_detectado,
    }