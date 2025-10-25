import cv2
import os
import numpy as np

def save_image(img, path):
    """
    Guarda una imagen RGB en disco (con creación de carpetas automática).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(path, img_bgr)
    if not success:
        raise IOError(f"❌ Error al guardar la imagen en: {path}")
    print(f"✅ Imagen guardada en: {path}")


def create_comparison_image(img1, img2, labels=["Original", "Envejecido"]):
    """
    Crea una imagen lado a lado para comparar dos imágenes.
    """
    # Asegurar tamaños iguales
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))

    comparison = np.hstack([img1_resized, img2_resized])

    # Añadir etiquetas visuales
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, labels[0], (20, 40), font, 1.0, (255, 255, 255), 2)
    cv2.putText(comparison, labels[1], (w + 20, 40), font, 1.0, (255, 255, 255), 2)

    return comparison


def validate_image_path(path):
    """Verifica que el archivo de imagen exista."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Imagen no encontrada: {path}")
    return True


def ensure_directory(path):
    """Crea un directorio si no existe."""
    os.makedirs(path, exist_ok=True)
