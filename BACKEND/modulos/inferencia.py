from ultralytics import YOLO
from typing import List, Union
import os

# ==================== CARGA DEL MODELO ====================
modelo = YOLO("BACKEND/modelo/MODELO.pt")  # Ruta al modelo YOLOv12n, asegúrate de tenerlo descargado

# ==================== FUNCIÓN: Inferencia sobre imagen única ====================
def inferir_imagen(ruta_imagen: str):
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"La ruta de la imagen no existe: {ruta_imagen}")
    resultados = modelo(ruta_imagen)
    print("INFERENCIA FINALIZADA")
    return resultados

# ==================== FUNCIÓN: Inferencia sobre múltiples imágenes (frames) ====================
def inferir_frames(frames: List[str]):
    resultados = []
    for frame in frames:
        resultados.append(modelo(frame))
    print("INFERENCIA FINALIZADA")
    return resultados

# ==================== FUNCIÓN UNIFICADORA: Inferencia automática ====================
def inferencia(entrada: Union[str, List[str]]):
    print("INFERENCIA INICIADA")
    if isinstance(entrada, str):
        return inferir_imagen(entrada)
    elif isinstance(entrada, list) and all(isinstance(x, str) for x in entrada):
        return inferir_frames(entrada)
    else:
        raise ValueError("Entrada no válida: debe ser una ruta (str) o lista de rutas (List[str])")