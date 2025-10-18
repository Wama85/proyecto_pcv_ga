import cv2
import numpy as np
import matplotlib.pyplot as plt
from ga import GeneticAlgorithm
from vision import load_image, detect_face_landmarks, apply_transformations
from fitness import evaluate_fitness
from utils import save_image

# --- CONFIGURACIÓN ---
IMG_PATH = "data/base/rostro.jpg"
GENERACIONES = 25
POBLACION = 30

# --- CARGAR IMAGEN BASE ---
base_img = load_image(IMG_PATH)
landmarks = detect_face_landmarks(base_img)

# --- LISTAS PARA GUARDAR EL PROGRESO ---
fitness_mejor = []
fitness_promedio = []

# --- DEFINIR GA ---
ga = GeneticAlgorithm(
    pop_size=POBLACION,
    generations=GENERACIONES,
    mutation_rate=0.2,
    crossover_rate=0.8,
    n_params=3  # [arrugas, brillo, saturación]
)

# --- FUNCIÓN DE FITNESS ---
def fitness_fn(individual):
    transformed = apply_transformations(base_img, individual)
    score = evaluate_fitness(transformed)
    return score

# --- EJECUTAR EVOLUCIÓN ---
best_params, best_fit = ga.run(fitness_fn)
print(f"\n✅ Mejor individuo encontrado: {best_params}")
print(f"🏆 Fitness final: {best_fit:.3f}")

# --- GUARDAR RESULTADO FINAL ---
final_img = apply_transformations(base_img, best_params)
save_image(final_img, "data/results/rostro_viejo.jpg")
print("📁 Imagen generada en data/results/rostro_viejo.jpg")

# --- MOSTRAR ANTES Y DESPUÉS ---
cv2.imshow("Rostro original", cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR))
cv2.imshow("Rostro envejecido", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
