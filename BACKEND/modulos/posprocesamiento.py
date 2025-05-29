import cv2
import numpy as np
import os
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

# --- Configuración global ---
CLASES = {
    0: 'Persona', 1: 'Oreja', 2: 'Orejeras', 3: 'Cara', 4: 'Protector-cara',
    5: 'Mascarilla', 6: 'Pie', 7: 'Herramienta', 8: 'Gafas', 9: 'Guantes',
    10: 'Casco', 11: 'Manos', 12: 'Cabeza', 13: 'Oreja', 14: 'Zapatos',
    15: 'Protector-seguridad', 16: 'Chaleco'
}

ICONOS_EPP = {
    'gris': {
        'Chaleco': 'svg/chaleco_gris.png',
        'Casco': 'svg/casco_gris.png',
        'Guantes': 'svg/guantes_gris.png',
        'Gafas': 'svg/gafas_gris.png'
    },
    'rojo': {
        'Chaleco': 'svg/chaleco_rojo.png',
        'Casco': 'svg/casco_rojo.png',
        'Guantes': 'svg/guantes_rojo.png',
        'Gafas': 'svg/gafas_rojo.png'
    }
}

ID_PERSONA = 0
TAM_ICONO = 30
ESPACIADO = 5

# --- Utilidades básicas ---

def calcular_iou(caja1, caja2):
    x1, y1, x2, y2 = caja1
    xa, ya, xb, yb = caja2
    inter_x1, inter_y1 = max(x1, xa), max(y1, ya)
    inter_x2, inter_y2 = min(x2, xb), min(y2, yb)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xb - xa) * (yb - ya)
    return inter_area / (area1 + area2 - inter_area)

def superponer_imagen_alpha(fondo, icono, x, y):
    fg, alpha = icono[..., :3], icono[..., 3:] / 255.0
    h, w = fg.shape[:2]
    roi = fondo[y:y+h, x:x+w]
    #fondo[y:y+h, x:x+w] = fg  # Dibujar icono opaco directamente
    fondo[y:y+h, x:x+w] = (1. - alpha) * roi + alpha * fg
    return fondo

# --- Evaluación de EPP ---

def evaluar_proteccion(personas, objetos_epp, clases, iou_min=0.3, margen=10):
    resultados = []
    for persona in personas:
        x1, y1, x2, y2 = [v - margen if i < 2 else v + margen for i, v in enumerate(persona)]
        estado = {"bbox": persona}
        for clase in clases:
            estado[clase] = any(
                (x1 <= (epp['bbox'][0] + epp['bbox'][2]) / 2 <= x2 and
                 y1 <= (epp['bbox'][1] + epp['bbox'][3]) / 2 <= y2) or
                calcular_iou([x1, y1, x2, y2], epp['bbox']) >= iou_min
                for epp in objetos_epp.get(clase, [])
            )
        resultados.append(estado)
    #print(f"[DEBUG] Evaluación de protección: {len(resultados)} personas evaluadas.")
    #print(f"[DEBUG] EVALUACIÓN: {resultados}")
    return resultados

# --- Posicionamiento de iconos ---

def posicion_valida(x, y, wv, wh, iw, hv, hh, ih, cajas, referencia, direccion):
    w, h = (wv, hv) if direccion in ['right', 'left'] else (wh, hh)
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        return False
    for x1, y1, x2, y2 in cajas:
        if [x1, y1, x2, y2] == referencia:
            continue
        if not (x + w < x1 or x > x2 or y + h < y1 or y > y2):
            return False
    return True

def calcular_posicion_iconos(bbox, todas_cajas, num_iconos, img_shape):
    x1, y1, x2, y2 = map(int, bbox)
    ih, iw = img_shape[:2]
    tamaño = TAM_ICONO + ESPACIADO
    wh, hh = tamaño * num_iconos, tamaño + ESPACIADO
    wv, hv = hh, wh

    #print(f"[DEBUG] BBox: {bbox}, Imagen: {iw}x{ih}")
    #print(f"[DEBUG] Tamaño horizontal: {wh}x{hh}, vertical: {wv}x{hv}")

    posiciones = {
        'top':    (x1, y1 - hh - ESPACIADO),
        'right':  (x2 + ESPACIADO, y1),
        'left':   (x1 - wv - ESPACIADO, y1),
        'bottom': (x1, y2 + ESPACIADO)
    }

    for dir, (px, py) in posiciones.items():
        #print(f"[DEBUG] Probando dirección '{dir}' en ({px}, {py})")
        if posicion_valida(px, py, wv, wh, iw, hv, hh, ih, todas_cajas, bbox, dir):
            w, h = (wv, hv) if dir in ['right', 'left'] else (wh, hh)
            orientacion = 'vertical' if dir in ['right', 'left'] else 'horizontal'
            #print(f"[DEBUG] Posición válida encontrada: {dir}, caja=({px}, {py}, {px + w}, {py + h})")
            return (px, py, px + w, py + h), orientacion

    # Fallback
    px = min(max(0, x1), iw - wh)
    py = max(0, y1 - hh)
    print(f"[WARN] No se encontró posición válida. Usando fallback en ({px}, {py})")
    #print(f"[DEBUG] Caja fallback: ({px}, {py}, {px + wh}, {py + hh})")
    return (px, py, px + wh, py + hh), 'horizontal'

# --- Dibujar iconos de EPP ---
def dibujar_iconos(imagen, caja_iconos, persona, clases, cumple, orientacion):
    x1, y1, x2, y2 = caja_iconos
    overlay = imagen.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    imagen = cv2.addWeighted(overlay, 0.4, imagen, 0.6, 0)
    #cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0) if cumple else (0, 0, 255), 2)

    x, y = x1 + ESPACIADO, y1 + ESPACIADO
    col = 0
    max_col = max(1, (x2 - x1) // (TAM_ICONO + ESPACIADO))
    #print(f"[DEBUG] Posición inicial de iconos: ({x}, {y}), Máximo columnas: {max_col}, Orientación: {orientacion}")

    for clase in clases:
        color = 'rojo' if not persona[clase] else 'gris'
        path = ICONOS_EPP[color].get(clase)

        #print(f"[DEBUG] Clase: {clase} - Estado: {'NO cumple' if color == 'rojo' else 'Cumple'}")
        #print(f"[DEBUG] Ruta del icono: {path}")

        if not path:
            print(f"[WARN] No hay icono definido para la clase: {clase}")
            continue

        if not os.path.exists(path):
            print(f"[ERROR] El archivo del icono no existe: {path}")
            continue

        icono = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if icono is None:
            print(f"[ERROR] No se pudo cargar el icono desde: {path}")
            continue

        h, w = icono.shape[:2]
        icono_redimensionado = cv2.resize(icono, (int(w * TAM_ICONO / h), TAM_ICONO), interpolation=cv2.INTER_AREA)
        imagen = superponer_imagen_alpha(imagen, icono_redimensionado, x, y)

        if orientacion == 'horizontal':
            col += 1
            x += TAM_ICONO + ESPACIADO
            if col >= max_col:
                x = x1 + ESPACIADO
                y += TAM_ICONO + ESPACIADO
        else:
            y += TAM_ICONO + ESPACIADO

    return imagen

from datetime import datetime

def generar_data_informe(evaluaciones, clases_usuario, timestamp, cam_id=0):
    data = []
    for idx, persona in enumerate(evaluaciones):
        fila = [timestamp, cam_id, idx]
        for clase in ['Gafas', 'Guantes', 'Casco', 'Chaleco']:
            if clase in clases_usuario:
                fila.append(bool(persona.get(clase, False)))
            else:
                fila.append("")
        data.append(fila)
    print(f"[DEBUG] Generado data {data}.")
    return data

# --- Función principal de posprocesamiento ---

def posprocesamiento(imagen, resultados, clases_usuario, zona,timestamp= datetime.now().strftime("%d/%m/%y_%H:%M:%S")):
    print("POSPROCESAMIENTO INICIADO")
    
    detecciones = resultados[0].boxes.data.cpu().numpy()
    personas_dentro = []
    personas_fuera = []
    objetos_epp = {cls: [] for cls in clases_usuario}

    
    print(f"[DEBUG] zona: {zona}")

    usar_zona = zona and len(zona) >= 3
    #print(f"[ANTENTÍSIMO] ZONA RELATIVA: {usar_zona}")
    # Convertir zona relativa a absoluta si los valores son <= 1
    zona_coords_absoluta = []
    #print(f"[ATENTÍSISISIMO] ZONA: {zona}")
    
 # --- Dibujo del polígono de zona ---
    if usar_zona:
        try:
            for punto in zona:
                x, y = punto
                if 0 <= x <= 1 and 0 <= y <= 1:
                    zona_coords_absoluta.append((x * 640, y * 640))
                else:
                    zona_coords_absoluta.append((x, y))
            print(f"[ATENTÍSISISISIMO] ZONA ABS: {zona_coords_absoluta}")
            polygon = Polygon(zona_coords_absoluta) if usar_zona else None
            puntos = np.array([[int(float(x)), int(float(y))] for x, y in zona_coords_absoluta], dtype=np.int32)
            puntos = puntos.reshape((-1, 1, 2))

            overlay = imagen.copy()
            cv2.fillPoly(overlay, [puntos], color=(0, 255, 255))  # Interior amarillo
            imagen = cv2.addWeighted(overlay, 0.3, imagen, 0.7, 0)  # Transparencia del relleno
            cv2.polylines(imagen, [puntos], isClosed=True, color=(0, 255, 255), thickness=2)  # Borde opaco

        except Exception as e:
            print(f"[ERROR] Zona inválida para dibujo: {e}")
            usar_zona = False

    for det in detecciones:
        x1, y1, x2, y2, conf, id_clase = det
        id_clase = int(id_clase)
        bbox = [x1, y1, x2, y2]
        clase = CLASES.get(id_clase, f"id:{id_clase}")
        #print(f"Detectado: {clase}")

        if id_clase == ID_PERSONA:
            centro_inferior = (int((x1 + x2) / 2), int(y2))
            cv2.circle(imagen, centro_inferior, 5, (255, 0, 0), -1)  # Azul
            if usar_zona:
                punto = Point(centro_inferior)
                if polygon.contains(punto):
                    personas_dentro.append(bbox)
                else:
                    personas_fuera.append(bbox)
            else:
                personas_dentro.append(bbox)

        elif clase in objetos_epp:
            objetos_epp[clase].append({'bbox': bbox, 'confidence': conf})

    evaluaciones = evaluar_proteccion(personas_dentro, objetos_epp, clases_usuario)
    faltantes = set()

    for persona in evaluaciones:
        #print(f"[DEBUG] Evaluando persona: {persona['bbox']}")
        x1, y1, x2, y2 = map(int, persona['bbox'])
        clases_faltantes = [cls for cls in clases_usuario if not persona[cls]]
        cumple = not clases_faltantes
        faltantes.update(clases_faltantes)

        if len(clases_usuario) == 0:
            print("[WARN] No hay clases de EPP definidas. Se omite dibujo de iconos.")
            continue

        overlay = imagen.copy()
        color = (0, 255, 0) if cumple else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        imagen = cv2.addWeighted(overlay, 0.25, imagen, 0.75, 0)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)

        caja_iconos, orientacion = calcular_posicion_iconos(persona['bbox'], [p['bbox'] for p in evaluaciones], len(clases_usuario), imagen.shape)
        #print(f"[DEBUG] Caja de iconos: {caja_iconos}, Orientación: {orientacion}")
        imagen = dibujar_iconos(imagen, caja_iconos, persona, clases_usuario, cumple, orientacion)

    for bbox in personas_fuera:
        x1, y1, x2, y2 = map(int, bbox)
        overlay = imagen.copy()
        color = (200, 200, 200)  # Verde: se "asume" cumplimiento
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        imagen = cv2.addWeighted(overlay, 0.25, imagen, 0.75, 0)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)

        persona_ficticia = {'bbox': bbox}
        for cls in clases_usuario:
            persona_ficticia[cls] = True

        #caja_iconos, orientacion = calcular_posicion_iconos(bbox, [p['bbox'] for p in evaluaciones], len(clases_usuario), imagen.shape)
        #imagen = dibujar_iconos(imagen, caja_iconos, persona_ficticia, clases_usuario, True, orientacion)
    
    
    # Obtener timestamp en formato requerido
    
    data_informe = generar_data_informe(evaluaciones, clases_usuario, timestamp)
    #print(f"[DEBUG] Data informe generado: {data_informe}")
    #print(f"[DEBUG] timestamp: {timestamp_actual}")


    print("POSPROCESAMIENTO FINALIZADO")
    return imagen, data_informe