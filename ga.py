import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, pop_size, generations, mutation_rate, crossover_rate,
                 n_params, elite_size=2):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_params = n_params
        self.elite_size = elite_size  # ✅ NUEVO: Elitismo configurable

    def init_population(self):
        """Inicialización de población"""
        return [np.random.rand(self.n_params) for _ in range(self.pop_size)]

    def select_tournament(self, pop, fitnesses, k=3):
        """Selección por torneo (k=3)"""
        participantes = random.sample(list(zip(pop, fitnesses)), k)
        participantes.sort(key=lambda x: x[1], reverse=True)
        return participantes[0][0]

    def crossover_uniform(self, p1, p2, prob=0.5):
        """Crossover uniforme"""
        if random.random() < self.crossover_rate:
            mask = np.random.rand(len(p1)) < prob
            child = np.where(mask, p1, p2)
            return child
        return p1.copy()

    def mutate(self, individual, generation, total_generations):
        """Mutación gaussiana adaptativa"""
        sigma = 0.1 * (1 - generation / total_generations)
        if random.random() < self.mutation_rate:
            individual += np.random.normal(0, sigma, self.n_params)
            individual = np.clip(individual, 0, 1)
        return individual

    def run(self, fitness_fn):
        """Ciclo principal del algoritmo"""
        pop = self.init_population()

        # ✅ NUEVO: Historial para graficar
        history = {
            'best': [],
            'avg': [],
            'worst': [],
            'std': []
        }

        for g in range(self.generations):
            # Evaluar población
            fitnesses = [fitness_fn(ind) for ind in pop]

            # Ranking global
            ranked = sorted(list(zip(pop, fitnesses)),
                            key=lambda x: x[1], reverse=True)
            elites = [p for p, f in ranked[:self.elite_size]]

            # ✅ NUEVO: Estadísticas detalladas
            best_fit = ranked[0][1]
            worst_fit = ranked[-1][1]
            avg_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)

            history['best'].append(best_fit)
            history['avg'].append(avg_fit)
            history['worst'].append(worst_fit)
            history['std'].append(std_fit)

            # ✅ MEJORADO: Mejor formato de salida
            progress = (g + 1) / self.generations * 100
            bar_length = 30
            filled = int(bar_length * (g + 1) / self.generations)
            bar = '█' * filled + '░' * (bar_length - filled)

            print(f"Gen {g+1:3d}/{self.generations} [{bar}] {progress:5.1f}% | "
                  f"Mejor: {best_fit:7.4f} | Promedio: {avg_fit:7.4f} | "
                  f"Std: {std_fit:6.4f}")

            # Nueva población
            new_pop = elites.copy()

            while len(new_pop) < self.pop_size:
                p1 = self.select_tournament(pop, fitnesses, k=3)
                p2 = self.select_tournament(pop, fitnesses, k=3)
                child = self.crossover_uniform(p1, p2, prob=0.5)
                child = self.mutate(child, g, self.generations)
                new_pop.append(child)

            pop = new_pop

        # Evaluar última generación
        fitnesses = [fitness_fn(ind) for ind in pop]
        best_idx = np.argmax(fitnesses)
        best_ind = pop[best_idx]
        best_fit = fitnesses[best_idx]

        print(f"\n🔥 Evolución finalizada. Mejor fitness obtenido: {best_fit:.4f}")

        # ✅ NUEVO: Devolver historial también
        return best_ind, best_fit, history