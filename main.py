import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Importar algoritmos
from ga import GeneticAlgorithm
from ga_improved import ImprovedGeneticAlgorithm
from ga_memetic import MemeticAlgorithm
from differential_evolution import DifferentialEvolution

# Importar módulos del sistema
from vision_optimized import load_image, detect_face_landmarks, apply_transformations
from fitness import evaluate_fitness
from utils import save_image, create_comparison_image, validate_image_path, ensure_directory


def parse_arguments():
    """Lee parámetros desde línea de comandos."""
    parser = argparse.ArgumentParser(description="🧬 Envejecimiento Facial con Algoritmos Evolutivos")
    parser.add_argument('--img', type=str, default='data/base/rostro4.jpg', help='Ruta de la imagen base')
    parser.add_argument('--generations', type=int, default=30, help='Número de generaciones')
    parser.add_argument('--population', type=int, default=40, help='Tamaño de la población')
    parser.add_argument('--mutation-rate', type=float, default=0.2, help='Tasa de mutación')
    parser.add_argument('--crossover-rate', type=float, default=0.8, help='Tasa de crossover')
    parser.add_argument('--output-dir', type=str, default='data/results', help='Carpeta de salida')
    parser.add_argument('--seed', type=int, default=42, help='Semilla para reproducibilidad')
    parser.add_argument('--algorithm', type=str, default='improved',
                        choices=['original', 'improved', 'memetic', 'de'],
                        help='Algoritmo: original, improved, memetic, de')
    return parser.parse_args()


def plot_fitness_evolution(f_best, f_avg, output_path):
    """Grafica la evolución del fitness."""
    plt.figure(figsize=(12, 6))
    generations = range(1, len(f_best) + 1)

    plt.plot(generations, f_best, 'b-', linewidth=2, label='Mejor Fitness')
    plt.plot(generations, f_avg, 'r--', linewidth=2, label='Fitness Promedio')

    plt.xlabel("Generación", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title("Evolución del Algoritmo Genético", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 Gráfica de evolución guardada en: {output_path}")


def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    print("=" * 70)
    print("🧬 SISTEMA DE ENVEJECIMIENTO FACIAL CON ALGORITMOS EVOLUTIVOS")
    print("=" * 70)

    validate_image_path(args.img)
    ensure_directory(args.output_dir)

    print(f"\n📸 Imagen de entrada: {args.img}")
    print(f"🤖 Algoritmo: {args.algorithm.upper()}")
    print(f"🔢 Generaciones: {args.generations}")
    print(f"👥 Población: {args.population}")
    print(f"🧬 Tasa de mutación: {args.mutation_rate}")
    print(f"🔀 Tasa de crossover: {args.crossover_rate}")

    # Cargar imagen
    print("\n⏳ Cargando imagen...")
    base_img = load_image(args.img)
    print(f"✅ Imagen cargada: {base_img.shape}")

    # Detección facial
    print("🔍 Detectando rostro...")
    landmarks = detect_face_landmarks(base_img)
    if isinstance(landmarks, np.ndarray):
        landmarks = tuple(landmarks)

    if landmarks == (0, 0, base_img.shape[1], base_img.shape[0]):
        print("⚠️  No se detectó rostro, se usará toda la imagen")
    else:
        x, y, w, h = landmarks
        print(f"✅ Rostro detectado en: x={x}, y={y}, w={w}, h={h}")

    # Seleccionar algoritmo
    print(f"\n🧬 Inicializando algoritmo: {args.algorithm.upper()}...")

    if args.algorithm == 'original':
        ga = GeneticAlgorithm(
            pop_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            n_params=4,
            elite_size=2
        )
    elif args.algorithm == 'improved':
        ga = ImprovedGeneticAlgorithm(
            pop_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            n_params=4,
            elite_size=3,
            diversity_threshold=0.01
        )
    elif args.algorithm == 'memetic':
        ga = MemeticAlgorithm(
            pop_size=args.population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            n_params=4,
            elite_size=3,
            local_search_prob=0.3
        )
    elif args.algorithm == 'de':
        ga = DifferentialEvolution(
            pop_size=args.population,
            generations=args.generations,
            F=0.8,
            CR=0.9,
            n_params=4
        )

    # Función de fitness
    def fitness_fn(individual):
        transformed = apply_transformations(base_img, individual)
        return evaluate_fitness(transformed)

    # Ejecutar evolución
    print("\n🚀 Iniciando evolución...\n")
    best_params, best_fit, history = ga.run(fitness_fn)

    print("\n" + "=" * 70)
    print("✅ EVOLUCIÓN COMPLETADA")
    print("=" * 70)
    print(f"\n🏆 Mejor individuo encontrado:")
    print(f"   • Wrinkle (arrugas):      {best_params[0]:.4f}")
    print(f"   • Desaturation (color):   {best_params[1]:.4f}")
    print(f"   • Fade (tonos cálidos):   {best_params[2]:.4f}")
    print(f"   • Edge strength (líneas): {best_params[3]:.4f}")
    print(f"\n📈 Fitness final: {best_fit:.4f}")

    # Generar imagen envejecida
    print("\n🎨 Generando rostro envejecido...")
    final_img = apply_transformations(base_img, best_params)
    aged_path = os.path.join(args.output_dir, "rostro_envejecido.jpg")
    save_image(final_img, aged_path)

    # Imagen de comparación
    print("🖼️  Creando imagen de comparación...")
    comparison = create_comparison_image(base_img, final_img)
    comparison_path = os.path.join(args.output_dir, "comparacion.jpg")
    save_image(comparison, comparison_path)

    # Gráfica de evolución
    print("📊 Generando gráfica de evolución...")
    plot_path = os.path.join(args.output_dir, "evolucion_fitness.png")
    plot_fitness_evolution(history['best'], history['avg'], plot_path)

    # Guardar parámetros óptimos
    params_path = os.path.join(args.output_dir, "parametros_optimos.txt")
    with open(params_path, "w", encoding="utf-8") as f:
        f.write("🔬 PARÁMETROS ÓPTIMOS ENCONTRADOS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Algoritmo: {args.algorithm.upper()}\n\n")
        f.write(f"Wrinkle (arrugas):      {best_params[0]:.6f}\n")
        f.write(f"Desaturation (color):   {best_params[1]:.6f}\n")
        f.write(f"Fade (tono cálido):     {best_params[2]:.6f}\n")
        f.write(f"Edge Strength (bordes): {best_params[3]:.6f}\n")
        f.write(f"\nFitness final: {best_fit:.6f}\n")
        f.write(f"\nConfiguración:\n")
        f.write(f"  Generaciones: {args.generations}\n")
        f.write(f"  Población: {args.population}\n")
        f.write(f"  Tasa mutación: {args.mutation_rate}\n")
        f.write(f"  Tasa crossover: {args.crossover_rate}\n")
    print(f"💾 Parámetros guardados en: {params_path}")

    # Mostrar resultados
    print("\n🖥️  Mostrando resultados...")
    cv2.imshow("Original", cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Envejecido", cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Comparación", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print("\n👉 Presiona cualquier tecla para cerrar las ventanas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("✨ PROCESO FINALIZADO EXITOSAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()