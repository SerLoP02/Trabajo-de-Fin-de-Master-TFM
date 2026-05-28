class NoSerieError(Exception):
    """Se lanza cuando no existe serie temporal disponible"""
    pass

class NullValuesError(BaseException):
    """Se lanza cuando existen valores nulos"""
    pass