import cv2
import numpy as np

def evaluate_fitness(img, weights=None):
    """
    Función de fitness CORREGIDA para evitar imágenes grises
    """
    if weights is None:
        weights = {
            'edges': 2.5,          # Arrugas
            'contrast': 1.0,       # Definición
            'brightness': -0.3,    # REDUCIDO (era -1.0)
            'saturation': 0.2,     # POSITIVO (era -1.0) - queremos MANTENER algo de color
            'texture': 1.2,        # Rugosidad
            'color_balance': 1.5   # NUEVO - penaliza exceso de gris
        }

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # 1. Bordes (arrugas)
    edges = cv2.Canny(gray, 70, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # 2. Saturación (QUEREMOS ALGO de saturación, no cero)
    saturation_mean = np.mean(hsv[..., 1]) / 255.0

    # Penalizar si es DEMASIADO baja (gris)
    if saturation_mean < 0.15:
        saturation_penalty = -2.0  #  MUY GRIS = MAL
    elif saturation_mean < 0.25:
        saturation_penalty = -0.5  # Un poco gris
    else:
        saturation_penalty = 0.0   #  Color adecuado

    # 3. Brillo
    brightness = np.mean(gray) / 255.0

    # Queremos reducción MODERADA, no extrema
    if brightness < 0.3:
        brightness_penalty = -1.5  #  Demasiado oscuro
    elif brightness > 0.7:
        brightness_penalty = -0.5  # Un poco claro
    else:
        brightness_penalty = 0.0   # Rango bueno

    # 4. Contraste
    contrast = np.std(gray) / 255.0

    # 5. Textura
    kernel_size = 5
    mean_filtered = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    mean_sq_filtered = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    variance = mean_sq_filtered - mean_filtered**2
    texture_score = np.mean(variance) / (255.0 ** 2)

    # 6. Balance de color (NUEVO)
    # Queremos que haya variación en los canales RGB
    b, g, r = cv2.split(img)
    color_std = (np.std(r) + np.std(g) + np.std(b)) / 3.0 / 255.0

    # Si todos los canales son iguales = gris = MAL
    channel_diff = np.abs(np.mean(r) - np.mean(b)) / 255.0
    if channel_diff < 0.05:
        color_balance = -1.0  #  Demasiado uniforme (gris)
    else:
        color_balance = channel_diff  # Hay diferencia de color

    # CÁLCULO FINAL
    fitness = (
            edge_density * weights['edges'] +
            contrast * weights['contrast'] +
            brightness * weights['brightness'] +
            saturation_mean * weights['saturation'] +
            texture_score * weights['texture'] +
            color_balance * weights['color_balance'] +
            saturation_penalty +
            brightness_penalty
    )

    return fitness


def evaluate_fitness_detailed(img):
    """Versión detallada para debugging"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    edges = cv2.Canny(gray, 70, 150)
    edge_density = np.sum(edges > 0) / edges.size
    saturation_mean = np.mean(hsv[..., 1]) / 255.0
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0

    kernel_size = 5
    mean_filtered = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    mean_sq_filtered = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    variance = mean_sq_filtered - mean_filtered**2
    texture_score = np.mean(variance) / (255.0 ** 2)

    b, g, r = cv2.split(img)
    color_std = (np.std(r) + np.std(g) + np.std(b)) / 3.0 / 255.0
    channel_diff = np.abs(np.mean(r) - np.mean(b)) / 255.0

    fitness = evaluate_fitness(img)

    return {
        'fitness_total': fitness,
        'edge_density': edge_density,
        'saturation': saturation_mean,
        'brightness': brightness,
        'contrast': contrast,
        'texture': texture_score,
        'color_std': color_std,
        'channel_diff': channel_diff
    }