import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from ga import GeneticAlgorithm
from vision import load_image, detect_face_landmarks, apply_transformations
from fitness import evaluate_fitness
from utils import (save_image, create_comparison_image,
                   validate_image_path, ensure_directory)

def parse_arguments():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Sistema de Envejecimiento Facial con AG')
    parser.add_argument('--img', type=str, default='data/base/rostro.jpeg',
                        help='Ruta de la imagen de entrada')
    parser.add_argument('--generations', type=int, default=25,
                        help='Número de generaciones')
    parser.add_argument('--population', type=int, default=30,
                        help='Tamaño de la población')
    parser.add_argument('--mutation-rate', type=float, default=0.2,
                        help='Tasa de mutación (0.0-1.0)')
    parser.add_argument('--crossover-rate', type=float, default=0.8,
                        help='Tasa de crossover (0.0-1.0)')
    parser.add_argument('--output-dir', type=str, default='data/results',
                        help='Directorio de salida')
    return parser.parse_args()


def plot_fitness_evolution(fitness_mejor, fitness_promedio, output_path):
    """
    Grafica la evolución del fitness a lo largo de las generaciones

    Args:
        fitness_mejor: Lista con el mejor fitness de cada generación
        fitness_promedio: Lista con el fitness promedio de cada generación
        output_path: Ruta donde guardar la gráfica
    """
    plt.figure(figsize=(12, 6))

    generations = range(1, len(fitness_mejor) + 1)

    plt.plot(generations, fitness_mejor, 'b-o', label='Mejor Fitness',
             linewidth=2, markersize=4)
    plt.plot(generations, fitness_promedio, 'r--s', label='Fitness Promedio',
             linewidth=2, markersize=4, alpha=0.7)

    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Evolución del Algoritmo Genético', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Anotación del mejor fitness final
    plt.annotate(f'Mejor: {fitness_mejor[-1]:.4f}',
                 xy=(len(fitness_mejor), fitness_mejor[-1]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"📊 Gráfica guardada en: {output_path}")
    plt.close()


def main():
    """Función principal"""
    # Parsear argumentos
    args = parse_arguments()

    print("=" * 60)
    print("🧬 SISTEMA DE ENVEJECIMIENTO FACIAL CON ALGORITMO GENÉTICO")
    print("=" * 60)

    # Validar y crear directorios
    try:
        validate_image_path(args.img)
        ensure_directory(args.output_dir)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"\n📁 Imagen de entrada: {args.img}")
    print(f"🔢 Generaciones: {args.generations}")
    print(f"👥 Población: {args.population}")
    print(f"🧬 Tasa de mutación: {args.mutation_rate}")
    print(f"🔀 Tasa de crossover: {args.crossover_rate}")

    # --- CARGAR IMAGEN BASE ---
    print("\n⏳ Cargando imagen...")
    base_img = load_image(args.img)
    print(f"✅ Imagen cargada: {base_img.shape}")

    # --- DETECTAR ROSTRO ---
    print("🔍 Detectando rostro...")
    landmarks = detect_face_landmarks(base_img)

    # ✅ CORRECCIÓN: Convertir a tupla si es array de NumPy
    if isinstance(landmarks, np.ndarray):
        landmarks = tuple(landmarks)

    # Verificar si se detectó rostro
    if landmarks == (0, 0, base_img.shape[1], base_img.shape[0]):
        print("⚠️  No se detectó rostro, se usará toda la imagen")
    else:
        x, y, w, h = landmarks
        print(f"✅ Rostro detectado en: x={x}, y={y}, w={w}, h={h}")

    # --- LISTAS PARA GUARDAR EL PROGRESO ---
    fitness_mejor = []
    fitness_promedio = []

    # --- DEFINIR GA ---
    print("\n🧬 Inicializando Algoritmo Genético...")
    ga = GeneticAlgorithm(
        pop_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        n_params=4  # [wrinkle, desaturation, fade_color, edge_strength]
    )

    # --- FUNCIÓN DE FITNESS ---
    def fitness_fn(individual):
        transformed = apply_transformations(base_img, individual)
        score = evaluate_fitness(transformed)
        return score

    # --- EJECUTAR EVOLUCIÓN ---
    print("\n🚀 Iniciando evolución...\n")
    best_params, best_fit, history = ga.run(fitness_fn)

    # Guardar historial
    fitness_mejor = history['best']
    fitness_promedio = history['avg']

    print("\n" + "=" * 60)
    print("✅ EVOLUCIÓN COMPLETADA")
    print("=" * 60)
    print(f"\n🏆 Mejor individuo encontrado:")
    print(f"   • Wrinkle (arrugas):      {best_params[0]:.4f}")
    print(f"   • Desaturation (color):   {best_params[1]:.4f}")
    print(f"   • Fade (tonos cálidos):   {best_params[2]:.4f}")
    print(f"   • Edge strength (líneas): {best_params[3]:.4f}")
    print(f"\n📈 Fitness final: {best_fit:.4f}")

    # --- GENERAR IMAGEN FINAL ---
    print("\n🎨 Generando imagen envejecida...")
    final_img = apply_transformations(base_img, best_params)

    # --- GUARDAR RESULTADO FINAL ---
    output_path = os.path.join(args.output_dir, "rostro_viejo.jpg")
    save_image(final_img, output_path)

    # --- CREAR IMAGEN DE COMPARACIÓN ---
    print("🖼️  Creando imagen de comparación...")
    comparison = create_comparison_image(base_img, final_img,
                                         ["Original", "Envejecido"])
    comparison_path = os.path.join(args.output_dir, "comparacion.jpg")
    save_image(comparison, comparison_path)

    # --- GRAFICAR EVOLUCIÓN ---
    print("📊 Generando gráfica de evolución...")
    plot_path = os.path.join(args.output_dir, "evolucion_fitness.png")
    plot_fitness_evolution(fitness_mejor, fitness_promedio, plot_path)

    # --- GUARDAR PARÁMETROS ---
    params_path = os.path.join(args.output_dir, "parametros_optimos.txt")
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write("PARÁMETROS ÓPTIMOS ENCONTRADOS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Wrinkle (arrugas):      {best_params[0]:.6f}\n")
        f.write(f"Desaturation (color):   {best_params[1]:.6f}\n")
        f.write(f"Fade (tonos cálidos):   {best_params[2]:.6f}\n")
        f.write(f"Edge strength (líneas): {best_params[3]:.6f}\n")
        f.write(f"\nFitness final: {best_fit:.6f}\n")
        f.write(f"\nGeneraciones: {args.generations}\n")
        f.write(f"Población: {args.population}\n")
    print(f"💾 Parámetros guardados en: {params_path}")

    # --- MOSTRAR ANTES Y DESPUÉS ---
    print("\n🖥️  Mostrando resultados...")
    cv2.imshow("Rostro original", cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Rostro envejecido", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Comparacion", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print("\n👉 Presiona cualquier tecla para cerrar las ventanas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("✨ PROCESO FINALIZADO EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    main()