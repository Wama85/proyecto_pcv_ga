import cv2
import numpy as np

def evaluate_fitness(img, weights=None):
    """
    Evalúa la calidad del envejecimiento facial

    Args:
        img: Imagen transformada (RGB)
        weights: Diccionario con pesos para cada métrica
                 Por defecto: {'edges': 2.0, 'contrast': 0.5,
                              'brightness': -1.0, 'saturation': -1.0}

    Returns:
        score: Valor de fitness (mayor = mejor envejecimiento)
    """
    # Pesos por defecto
    if weights is None:
        weights = {
            'edges': 2.0,        # Más arrugas = mejor
            'contrast': 0.5,     # Más definición = mejor
            'brightness': -1.0,  # Menos brillo = más viejo
            'saturation': -1.0,  # Menos color = más viejo
            'texture': 1.0       # ✅ NUEVO: Más textura = más realista
        }

    # Conversiones de espacio de color
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # ===== MÉTRICA 1: Densidad de bordes (arrugas) =====
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.sum(edges > 0) / edges.size

    # ===== MÉTRICA 2: Saturación promedio =====
    saturation_mean = np.mean(hsv[..., 1]) / 255.0

    # ===== MÉTRICA 3: Brillo promedio =====
    brightness = np.mean(gray) / 255.0

    # ===== MÉTRICA 4: Contraste =====
    contrast = np.std(gray) / 255.0

    # ===== MÉTRICA 5: Textura (varianza local) ✅ NUEVO =====
    # Calcula la varianza en ventanas de 5x5
    kernel_size = 5
    mean_filtered = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    mean_sq_filtered = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    variance = mean_sq_filtered - mean_filtered**2
    texture_score = np.mean(variance) / (255.0 ** 2)

    # ===== CÁLCULO DE FITNESS =====
    fitness = (
            edge_density * weights['edges'] +
            contrast * weights['contrast'] +
            brightness * weights['brightness'] +
            saturation_mean * weights['saturation'] +
            texture_score * weights['texture']
    )

    return fitness


def evaluate_fitness_detailed(img):
    """
    Versión detallada que devuelve todas las métricas individuales
    Útil para análisis y debugging

    Returns:
        dict: Diccionario con todas las métricas
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.sum(edges > 0) / edges.size
    saturation_mean = np.mean(hsv[..., 1]) / 255.0
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0

    # Textura
    kernel_size = 5
    mean_filtered = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    mean_sq_filtered = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    variance = mean_sq_filtered - mean_filtered**2
    texture_score = np.mean(variance) / (255.0 ** 2)

    fitness = (edge_density * 2) + (contrast * 0.5) - (brightness + saturation_mean) + texture_score

    return {
        'fitness_total': fitness,
        'edge_density': edge_density,
        'saturation': saturation_mean,
        'brightness': brightness,
        'contrast': contrast,
        'texture': texture_score
    }