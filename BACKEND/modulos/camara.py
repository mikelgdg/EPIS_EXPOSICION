# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 17:43:52 2025

@author: EXTaviton
"""

import cv2
import time
import uuid
import os
from BACKEND.modulos.inferencia import inferencia
from BACKEND.modulos.posprocesamiento import posprocesamiento
from BACKEND.modulos.config import carpeta_salidas



def get_frame(clases_permitidas=None, grabar=False,zona=[]):
    video = cv2.VideoCapture(0)
    nombre_archivo = f"{carpeta_salidas}/deteccion_vivo_{uuid.uuid4().hex[:8]}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    prev_time = time.time()
    frame_times = []  # Para calcular FPS promedio

    while True:
        success, frame = video.read()
        if not success:
            print("DEBUG: No se pudo leer el frame de la cámara.")
            break

        # Tiempo actual y delay desde el último frame
        current_time = time.time()
        elapsed = current_time - prev_time
        prev_time = current_time
        frame_times.append(elapsed)

        # Redimensionar el frame
        frame = cv2.resize(frame, (640, 640))

        # Guardar el frame como un archivo temporal
        temp_filepath = f"{carpeta_salidas}/temp_frame_{uuid.uuid4().hex[:8]}.jpg"
        cv2.imwrite(temp_filepath, frame)

        # Procesar el frame con inferencia
        try:
            resultados = inferencia(temp_filepath)
            processed_frame, faltantes=posprocesamiento(frame, resultados, clases_permitidas,zona)
            frame_detectado = processed_frame.copy()
        except Exception as e:
            frame_detectado = frame  # Si falla, usar el frame original

        # Eliminar el archivo temporal
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        # Grabar el frame procesado si es necesario
        if grabar:
            if out is None:
                height, width = frame_detectado.shape[:2]
                out = cv2.VideoWriter(nombre_archivo, fourcc, 2.0, (width, height))
            out.write(frame_detectado)

        # Codificar el frame para enviar al navegador
        ret, jpeg = cv2.imencode(".jpg", frame_detectado)
        if not ret:
            print("DEBUG: Error al codificar el frame, omitiendo.")
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n")

    video.release()
    if out:
        out.release()

        # Calcular FPS promedio
        if len(frame_times) > 0:
            fps_real = 1 / (sum(frame_times) / len(frame_times))
            print(f"FPS real calculado: {fps_real:.2f}")
            ajustar_fps(nombre_archivo, fps_real)

def ajustar_fps(video_path, fps_deseado):
    """Recodifica el video a la velocidad deseada"""
    temp_path = video_path.replace(".mp4", "_temp.mp4")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_path, fourcc, fps_deseado, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    os.remove(video_path)
    os.rename(temp_path, video_path)
    print(f"Video recodificado a {fps_deseado:.2f} FPS correctamente.")