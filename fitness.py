import cv2
import numpy as np

def evaluate_fitness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    brightness = np.mean(gray) / 255.0
    smoothness = 1.0 - np.std(gray) / 128.0  # mide ruido

    # Ideal: más bordes, brillo medio y sin ruido
    score = (edge_density * 2.0) - abs(brightness - 0.5) - (1 - smoothness)
    return score
