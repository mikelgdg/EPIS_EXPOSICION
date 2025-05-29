from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os, json, re
import cv2
import uuid
import datetime
from pymediainfo import MediaInfo

from BACKEND.modulos.config import carpeta_subidas, extensiones_admitidas, carpeta_salidas
from BACKEND.modulos.preprocesamiento import preprocesamiento
from BACKEND.modulos.inferencia import inferencia
from BACKEND.modulos.posprocesamiento import posprocesamiento
from BACKEND.modulos.camara import get_frame


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/salidas", StaticFiles(directory=os.path.abspath("salidas")), name="salidas")
data_informe=[]

def secure_filename(filename):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)

from pymediainfo import MediaInfo
import datetime
import os

def get_timestamps(file_path):
    media_info = MediaInfo.parse(file_path)
    start_time = None
    frame_rate = None
    frame_count = None
    is_video = False

    # Buscar datos en los metadatos
    for track in media_info.tracks:
        if track.track_type == "Video":
            is_video = True
            frame_rate = float(track.frame_rate) if track.frame_rate else None
            frame_count = int(track.frame_count) if track.frame_count else None
        if track.track_type in ("Video", "Image"):
            raw_date = track.encoded_date or track.tagged_date
            if raw_date:
                # Ejemplo de formato: 'UTC 2025-05-17 11:23:45'
                raw_date = raw_date.replace("UTC ", "")
                try:
                    start_time = datetime.datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"[ERROR] Formato de fecha no reconocido: {raw_date}")
                    continue

    # Si no se pudo obtener start_time, usar fecha del sistema de archivos como fallback
    if not start_time:
        ts = os.path.getmtime(file_path)
        start_time = datetime.datetime.fromtimestamp(ts)

    # Si es video, generar timestamps por frame
    if is_video and frame_rate:
        # Si no hay frame_count, asumir duración con duración en milisegundos
        if not frame_count:
            for track in media_info.tracks:
                if track.track_type == "Video" and track.duration:
                    duration_s = float(track.duration) / 1000
                    frame_count = int(duration_s * frame_rate)
                    break

        timestamps = [start_time + datetime.timedelta(seconds=i / frame_rate) for i in range(frame_count or 0)]
    else:
        # Imagen o video sin datos suficientes
        timestamps = [start_time]

    # Devolver los timestamps con precisión de milisegundos
    formatted = [ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] for ts in timestamps]
    return formatted

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), clases: str = Form(...), zona: str = Form(...)):
    print(f"Nombre del archivo: {file.filename}")
    print(f"Tipo: {file.content_type}")
    print(f"Clases: {clases}")
    print(f"Zona: {zona}")
    try:
        # Asegura nombre y extensión
        filename = secure_filename(file.filename)
        extension = filename.rsplit(".", 1)[1].lower()
        if extension not in extensiones_admitidas:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")
        
        os.makedirs(carpeta_subidas, exist_ok=True)
        os.makedirs(carpeta_salidas, exist_ok=True)

        # Guarda archivo subido
        filepath = os.path.join(carpeta_subidas, filename)
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)

        clases_permitidas = json.loads(clases)
        zona_coords = json.loads(zona)

        if extension == "json":
            print("JSON SUBIDO!!!!")
            try:
                datos = json.loads(contents)
                objects = datos.get("objects", [])

                coords = []
                for obj in objects:
                    if obj.get("type") == "path":
                        coords = [(cmd[1], 200 - cmd[2]) for cmd in obj["path"] if cmd[0] in ("M", "L")]
                        if len(coords) >= 3:
                            print("¡Polígono finalizado!")
                            print("Coordenadas:", coords)
                            break
                        else:
                            print("Dibuja al menos 3 puntos para formar el polígono.")

                if coords:
                    with open("zona_temporal.json", "w") as f:
                        json.dump(coords, f)
                    print("Zona guardada exitosamente:", coords)
                    return {"mensaje": "Zona cargada correctamente", "coordenadas": coords}
                else:
                    raise HTTPException(status_code=400, detail="No se detectó un polígono válido.")
            except Exception as e:
                print(f"[ERROR procesando JSON] {e}")
                raise HTTPException(status_code=500, detail="Error al procesar el archivo JSON.")

        elif extension in ["jpg", "jpeg", "png"]:
            if not zona_coords and os.path.exists("zona_temporal.json"):
                try:
                    with open("zona_temporal.json", "r") as f:
                        zona_coords = json.load(f)
                except Exception as e:
                    print(f"[ERROR cargando zona]: {e}")
                    zona_coords = []

            # --- Desempaquetar si viene como [[[x, y], ...]] ---
            if isinstance(zona_coords, list) and len(zona_coords) == 1 and isinstance(zona_coords[0], list):
                zona_coords = zona_coords[0]

            # --- Convertir listas a tuplas ---
            zona_coords = [tuple(p) for p in zona_coords]

            print(f"ZONA: {zona_coords}")

            ruta_imagen_preprocesada = os.path.join(carpeta_subidas, f"resized_{filename}")
            ts_list = get_timestamps(filepath)
            imagen_preprocesada = preprocesamiento(filepath, ruta_imagen_preprocesada, es_video=False)
            resultados = inferencia(ruta_imagen_preprocesada)
            imagen_posprocesada, data_informe = posprocesamiento(imagen_preprocesada, resultados, clases_permitidas, zona_coords)
            print(f"Data informe: {data_informe}")
            ruta_imagen_posprocesada = os.path.join(carpeta_salidas, f"resultado_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(ruta_imagen_posprocesada, imagen_posprocesada)

            if os.path.exists("zona_temporal.json"):
                os.remove("zona_temporal.json")

            return {
                "ruta_resultado": ruta_imagen_posprocesada,
                "nombre_resultado": os.path.basename(ruta_imagen_posprocesada),
                "data_informe": data_informe
            }



        elif extension == "mp4":
            temp_frames_dir = os.path.join(carpeta_subidas, f"resized_frames_{uuid.uuid4().hex}")
            os.makedirs(temp_frames_dir, exist_ok=True)

            ts_list = get_timestamps(filepath)
            for ts in ts_list:
                print(ts)
            
            # --- Desempaquetar si viene como [[[x, y], ...]] ---
            if isinstance(zona_coords, list) and len(zona_coords) == 1 and isinstance(zona_coords[0], list):
                zona_coords = zona_coords[0]

            # --- Convertir listas a tuplas ---
            zona_coords = [tuple(p) for p in zona_coords]


            frames_preprocesados = preprocesamiento(filepath, temp_frames_dir, es_video=True)
            resultados = inferencia(frames_preprocesados)

            rutas_frames_resultado = []
            data_informe_completa = []

            for i, (frame_path, resultado) in enumerate(zip(frames_preprocesados, resultados)):
                #print(f"RESULTADOS: {resultados}")
                imagen = cv2.imread(frame_path)
                imagen_posprocesada, data_informe = posprocesamiento(imagen, resultado, clases_permitidas,zona_coords,ts_list[i])
                print(f"Data informe: {data_informe}")
                data_informe_completa.append(data_informe)

                nombre_frame_resultado = f"frame_proc_{i:04d}.jpg"
                ruta_frame_resultado = os.path.join(carpeta_salidas, nombre_frame_resultado)
                cv2.imwrite(ruta_frame_resultado, imagen_posprocesada)
                rutas_frames_resultado.append(nombre_frame_resultado)

            print(rutas_frames_resultado)
            return {
                "frames_resultados": rutas_frames_resultado,
                "carpeta_resultados": carpeta_salidas,
                "data_informe_completa": data_informe_completa
            }

        else:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail="Error procesando el archivo.")
    


@app.get("/salidas/{filename}")
async def serve_result_image(filename: str):
    file_path = os.path.join(carpeta_salidas, filename)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
from fastapi.responses import FileResponse

from fastapi.responses import StreamingResponse
from starlette.requests import Request

from fastapi import Query

@app.get("/video_feed/")
async def video_feed(clases: str = Query(default=""), zona: str = Query(default="[]")):
    ### Endpoint para proporcionar un feed de video en tiempo real.
    ### -> Utiliza StreamingResponse para enviar frames de video.
   
    # Devuelve un stream de video en tiempo real
    clases_filtradas = clases.split(",") if clases else []
    return StreamingResponse(get_frame(clases_filtradas), media_type="multipart/x-mixed-replace; boundary=frame")


from fastapi import Query

@app.get("/video_feed_raw/")
async def video_feed(clases: str = Query(default=""), zona: str = Query(default="[]")):
    ### Endpoint para proporcionar un feed de video en tiempo real.
    ### -> Utiliza StreamingResponse para enviar frames de video.
   
    # Devuelve un stream de video en tiempo real
    clases_filtradas = clases.split(",") if clases else []
    zona_coords = json.loads(zona)
    return StreamingResponse(get_frame(clases_filtradas, zona_coords), media_type="multipart/x-mixed-replace; boundary=frame")
