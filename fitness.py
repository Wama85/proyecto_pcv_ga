import cv2
import numpy as np

def evaluate_fitness(img, weights=None):
    """
    Función de fitness CORREGIDA para envejecimiento NATURAL

    El problema anterior: penalizaba demasiado la saturación → imágenes grises
    Solución: Balancear para buscar envejecimiento VISIBLE pero NATURAL
    """
    if weights is None:
        weights = {
            'edges': 2.0,           # Queremos arrugas
            'contrast': 1.2,        # Queremos definición
            'brightness': -0.4,     # Queremos ALGO de reducción (no mucha)
            'saturation': -0.3,     # Queremos POCA reducción (mantener color)
            'texture': 1.5,         # Queremos textura
            'color_warmth': 1.0,    # NUEVO: Queremos tonos cálidos
            'gray_penalty': -3.0    # NUEVO: Penalizar MUCHO si es gris
        }

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rgb = img

    # ===============================
    # 1. BORDES (arrugas)
    # ===============================
    edges = cv2.Canny(gray, 70, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # ===============================
    # 2. SATURACIÓN (mantener ALGO de color)
    # ===============================
    saturation_mean = np.mean(hsv[..., 1]) / 255.0

    # ⚠️ PENALIZACIÓN si es DEMASIADO gris
    if saturation_mean < 0.20:
        gray_penalty = weights['gray_penalty']  # -3.0 = MUY MAL
    elif saturation_mean < 0.30:
        gray_penalty = -1.0  # Algo malo
    else:
        gray_penalty = 0.0  # OK

    # ===============================
    # 3. BRILLO (reducción moderada)
    # ===============================
    brightness = np.mean(gray) / 255.0

    # ===============================
    # 4. CONTRASTE
    # ===============================
    contrast = np.std(gray) / 255.0

    # ===============================
    # 5. TEXTURA
    # ===============================
    kernel_size = 5
    mean_filtered = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
    mean_sq_filtered = cv2.blur((gray.astype(float))**2, (kernel_size, kernel_size))
    variance = mean_sq_filtered - mean_filtered**2
    texture_score = np.mean(variance) / (255.0 ** 2)

    # ===============================
    # 6. CALIDEZ DE COLOR (NUEVO)
    # ===============================
    # Queremos que haya más rojo/amarillo que azul (piel envejecida)
    b, g, r = cv2.split(rgb)

    warmth = (np.mean(r) - np.mean(b)) / 255.0

    # Si warmth > 0 = más cálido (BUENO)
    # Si warmth < 0 = más frío/azul (MALO)
    if warmth < -0.05:
        warmth_score = -1.0  # Demasiado azul
    else:
        warmth_score = warmth  # Cálido = bueno

    # ===============================
    # 7. EVITAR IMÁGENES UNIFORMES
    # ===============================
    # Si todos los canales RGB son muy similares = gris
    channel_variation = np.std([np.mean(r), np.mean(g), np.mean(b)]) / 255.0

    if channel_variation < 0.02:
        uniformity_penalty = -2.0  # MUY uniforme = gris = MAL
    else:
        uniformity_penalty = 0.0

    # ===============================
    # CÁLCULO FINAL DEL FITNESS
    # ===============================
    fitness = (
            edge_density * weights['edges'] +
            contrast * weights['contrast'] +
            brightness * weights['brightness'] +
            saturation_mean * weights['saturation'] +
            texture_score * weights['texture'] +
            warmth_score * weights['color_warmth'] +
            gray_penalty +
            uniformity_penalty
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
    warmth = (np.mean(r) - np.mean(b)) / 255.0
    channel_variation = np.std([np.mean(r), np.mean(g), np.mean(b)]) / 255.0

    fitness = evaluate_fitness(img)

    return {
        'fitness_total': fitness,
        'edge_density': edge_density,
        'saturation': saturation_mean,
        'brightness': brightness,
        'contrast': contrast,
        'texture': texture_score,
        'warmth': warmth,
        'channel_variation': channel_variation
    }