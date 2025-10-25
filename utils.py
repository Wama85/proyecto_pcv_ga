import cv2
import os
import numpy as np

def save_image(img, path):
    """
    Guarda una imagen RGB en formato BGR (OpenCV)

    Args:
        img: Imagen en formato RGB (numpy array)
        path: Ruta donde guardar la imagen
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convertir de RGB a BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Guardar
    success = cv2.imwrite(path, img_bgr)

    if not success:
        raise IOError(f"Error al guardar la imagen en: {path}")

    print(f"✅ Imagen guardada en: {path}")


def create_comparison_image(img1, img2, labels=["Original", "Transformada"]):
    """
    Crea una imagen lado a lado para comparar dos imágenes

    Args:
        img1: Primera imagen (RGB)
        img2: Segunda imagen (RGB)
        labels: Etiquetas para cada imagen

    Returns:
        Imagen combinada
    """
    # Asegurar que ambas imágenes tengan el mismo tamaño
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Usar el tamaño más pequeño
    h = min(h1, h2)
    w = min(w1, w2)

    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))

    # Concatenar horizontalmente
    comparison = np.hstack([img1_resized, img2_resized])

    # Agregar texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, labels[0], (20, 40), font, 1, (255, 255, 255), 2)
    cv2.putText(comparison, labels[1], (w + 20, 40), font, 1, (255, 255, 255), 2)

    return comparison


def validate_image_path(path):
    """Valida que la imagen exista"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Imagen no encontrada: {path}")
    return True


def ensure_directory(path):
    """Crea directorio si no existe"""
    os.makedirs(path, exist_ok=True)