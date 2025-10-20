import cv2
import numpy as np

def evaluate_fitness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Bordes (más = más envejecido)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.sum(edges > 0) / edges.size

    # Saturación (menos = más viejo)
    saturation_mean = np.mean(hsv[..., 1]) / 255.0

    # Brillo medio (menos = más viejo)
    brightness = np.mean(gray) / 255.0

    # Contraste (ligero incremento = más interesante)
    contrast = np.std(gray) / 255.0

    # Fitness: más bordes, más contraste, menos brillo y saturación
    fitness = (edge_density * 2) + (contrast * 0.5) - (brightness + saturation_mean)
    return fitness
