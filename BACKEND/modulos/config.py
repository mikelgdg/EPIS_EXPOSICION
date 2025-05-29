import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
carpeta_subidas = os.path.abspath(os.path.join(BASE_DIR, "../../subidas"))
carpeta_salidas = os.path.abspath(os.path.join(BASE_DIR, "../../salidas"))
extensiones_admitidas = ["jpg", "jpeg", "png", "mp4", "json"]