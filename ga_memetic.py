import numpy as np
import random
from scipy.optimize import minimize

class MemeticAlgorithm:
    """
    Algoritmo Memético = Algoritmo Genético + Búsqueda Local
    Combina exploración global (AG) con explotación local (Hill Climbing)
    """
    def __init__(self, pop_size, generations, mutation_rate, crossover_rate,
                 n_params, elite_size=3, local_search_prob=0.3):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_params = n_params
        self.elite_size = elite_size
        self.local_search_prob = local_search_prob  # Probabilidad de búsqueda local

    def init_population(self):
        return [np.random.rand(self.n_params) for _ in range(self.pop_size)]

    def select_tournament(self, pop, fitnesses, k=5):
        participantes = random.sample(list(zip(pop, fitnesses)), k)
        participantes.sort(key=lambda x: x[1], reverse=True)
        return participantes[0][0]

    def crossover_blx(self, p1, p2, alpha=0.5):
        """BLX-α crossover"""
        if random.random() < self.crossover_rate:
            child = np.zeros(self.n_params)
            for i in range(self.n_params):
                cmin = min(p1[i], p2[i])
                cmax = max(p1[i], p2[i])
                range_val = cmax - cmin

                lower = cmin - alpha * range_val
                upper = cmax + alpha * range_val

                child[i] = random.uniform(max(0, lower), min(1, upper))

            return child
        return p1.copy()

    def mutate(self, individual, generation, total_generations):
        """Mutación adaptativa"""
        sigma = 0.15 * np.exp(-3 * generation / total_generations)
        if random.random() < self.mutation_rate:
            individual += np.random.normal(0, sigma, self.n_params)
            individual = np.clip(individual, 0, 1)
        return individual

    def local_search(self, individual, fitness_fn, max_iterations=10):
        """
        Búsqueda local usando Nelder-Mead
        Refina la solución encontrada por el AG
        """
        def objective(x):
            # Negativo porque minimize busca mínimo
            return -fitness_fn(np.clip(x, 0, 1))

        # Búsqueda local con límites
        bounds = [(0, 1) for _ in range(self.n_params)]

        try:
            result = minimize(
                objective,
                individual,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations, 'disp': False}
            )

            if result.success:
                return result.x, -result.fun
            else:
                return individual, fitness_fn(individual)
        except:
            return individual, fitness_fn(individual)

    def run(self, fitness_fn):
        """Ejecuta el algoritmo memético"""
        pop = self.init_population()
        history = {'best': [], 'avg': [], 'local_searches': []}

        local_search_count = 0

        for g in range(self.generations):
            fitnesses = [fitness_fn(ind) for ind in pop]
            ranked = sorted(list(zip(pop, fitnesses)), key=lambda x: x[1], reverse=True)
            elites = [p for p, f in ranked[:self.elite_size]]

            # Estadísticas
            best_fit = ranked[0][1]
            avg_fit = np.mean(fitnesses)

            history['best'].append(best_fit)
            history['avg'].append(avg_fit)

            # Progreso
            progress = (g + 1) / self.generations * 100
            bar_len = 30
            filled = int(bar_len * (g + 1) / self.generations)
            bar = '█' * filled + '░' * (bar_len - filled)

            print(f"Gen {g+1:3d}/{self.generations} [{bar}] {progress:5.1f}% | "
                  f"Mejor: {best_fit:7.4f} | Prom: {avg_fit:7.4f} | "
                  f"BL: {local_search_count}")

            # Nueva generación
            new_pop = elites.copy()

            while len(new_pop) < self.pop_size:
                p1 = self.select_tournament(pop, fitnesses)
                p2 = self.select_tournament(pop, fitnesses)
                child = self.crossover_blx(p1, p2)
                child = self.mutate(child, g, self.generations)

                # Búsqueda local probabilística
                if random.random() < self.local_search_prob:
                    child, _ = self.local_search(child, fitness_fn, max_iterations=5)
                    local_search_count += 1

                new_pop.append(child)

            pop = new_pop
            history['local_searches'].append(local_search_count)

        # Final
        fitnesses = [fitness_fn(ind) for ind in pop]
        best_idx = np.argmax(fitnesses)
        best_ind = pop[best_idx]
        best_fit = fitnesses[best_idx]

        # Búsqueda local final intensiva
        print("\n🔍 Realizando búsqueda local final intensiva...")
        best_ind, best_fit = self.local_search(best_ind, fitness_fn, max_iterations=50)

        print(f"\n🔥 Evolución finalizada. Mejor fitness: {best_fit:.4f}")
        print(f"🔍 Búsquedas locales realizadas: {local_search_count}")

        return best_ind, best_fit, history