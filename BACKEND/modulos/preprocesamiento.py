import os
import cv2
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np


TAMANIO_OBJETIVO = (640, 640)
FPS_OBJETIVO = 4

def preprocesar_imagen(ruta_entrada, ruta_salida=None, tam=(640, 640)):
    img_bgr = cv2.imread(ruta_entrada)
    if img_bgr is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta_entrada}")
    img_resized = cv2.resize(img_bgr, tam)
    if ruta_salida:
        cv2.imwrite(ruta_salida, img_resized)

    print("PREPROCESAMIENTO FINALIZADO")
    return img_resized # Devuelve la imagen redimensionada

def preprocesar_video(ruta_video, carpeta_salida, tam=TAMANIO_OBJETIVO, fps_objetivo=FPS_OBJETIVO):
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise Exception(f"No se pudo abrir el video: {ruta_video}")

    os.makedirs(carpeta_salida, exist_ok=True)

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    intervalo = max(1, int(round(fps_original / fps_objetivo)))
    rutas_frames = []
    indice = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if indice % intervalo == 0:
            frame_redim = cv2.resize(frame, tam, interpolation=cv2.INTER_AREA)
            ruta_frame = os.path.join(carpeta_salida, f"frame_{indice:04d}.jpg")
            cv2.imwrite(ruta_frame, frame_redim)
            rutas_frames.append(ruta_frame)
        indice += 1

    cap.release()
    print("PREPROCESAMIENTO FINALIZADO")
    return rutas_frames #una lista con cada ruta de los frames




def preprocesamiento(entrada, salida, es_video=False):
    print("PREPROCESAMIENTO INICIADO")
    if es_video:
        return preprocesar_video(entrada, salida) #lista con cada ruta de los frames
    else:
        return preprocesar_imagen(entrada, salida)