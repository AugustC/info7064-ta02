import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageOps

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif")

def list_images(directory):
    return sorted(p for p in glob.glob(os.path.join(directory, "*"))
                  if p.lower().endswith(EXT))

def load_image(path):
    # PIL + exif_transpose: respeita a orientação gravada pelo celular
    if path.lower().endswith((".heic", ".heif", ".jpg", ".jpeg")):
        img = ImageOps.exif_transpose(Image.open(path))
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.imread(path, cv2.IMREAD_COLOR)

def upload_images(directory):
    pass

def save_corrected_images():
    pass

def pair_stereo_images(left_dir, right_dir):
    left_images = list_images(left_dir)
    right_images = list_images(right_dir)
    paired_images = []
    for left_path in left_images:
        filename = os.path.basename(left_path)
        right_path = os.path.join(right_dir, filename)
        if os.path.exists(right_path):
            paired_images.append((left_path, right_path))
    return paired_images

#arquivos = listar(PASTA_CALIB)
#testes = listar(PASTA_TESTE)
#print(len(arquivos), "fotos de calibração |", len(testes), "de teste")
#
#TAM = carregar(arquivos[0]).shape[:2][::-1]         # (largura, altura)
#print("resolução:", TAM)
