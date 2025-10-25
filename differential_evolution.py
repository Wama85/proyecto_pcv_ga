import numpy as np

class DifferentialEvolution:
    """
    Evolución Diferencial (DE)
    Algoritmo diferente al GA, específicamente diseñado para optimización continua
    Muy efectivo para espacios de parámetros reales
    """
    def __init__(self, pop_size, generations, F=0.8, CR=0.9, n_params=4):
        """
        Args:
            pop_size: Tamaño de población
            generations: Número de generaciones
            F: Factor de escala diferencial (0.5-1.0)
            CR: Tasa de crossover (0.8-1.0)
            n_params: Número de parámetros
        """
        self.pop_size = pop_size
        self.generations = generations
        self.F = F  # Factor de mutación diferencial
        self.CR = CR  # Crossover rate
        self.n_params = n_params

    def init_population(self):
        """Inicializa población aleatoria"""
        return np.random.rand(self.pop_size, self.n_params)

    def mutate_de(self, pop, best_idx, i):
        """
        Mutación diferencial: v = x_best + F * (x_r1 - x_r2)
        Utiliza la diferencia entre individuos para guiar la búsqueda
        """
        # Seleccionar 3 índices aleatorios diferentes de i
        candidates = list(range(self.pop_size))
        candidates.remove(i)

        r1, r2 = np.random.choice(candidates, 2, replace=False)

        # Mutación diferencial
        mutant = pop[best_idx] + self.F * (pop[r1] - pop[r2])

        return np.clip(mutant, 0, 1)

    def crossover_de(self, target, mutant):
        """
        Crossover binomial
        Combina el individuo objetivo con el mutante
        """
        child = target.copy()

        # Garantizar al menos un componente del mutante
        j_rand = np.random.randint(0, self.n_params)

        for j in range(self.n_params):
            if np.random.rand() < self.CR or j == j_rand:
                child[j] = mutant[j]

        return child

    def run(self, fitness_fn):
        """Ejecuta la evolución diferencial"""
        # Inicializar
        pop = self.init_population()
        fitnesses = np.array([fitness_fn(ind) for ind in pop])

        history = {'best': [], 'avg': [], 'worst': [], 'std': []}

        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_individual = pop[best_idx].copy()

        for g in range(self.generations):
            # Para cada individuo
            for i in range(self.pop_size):
                # Mutación diferencial
                mutant = self.mutate_de(pop, best_idx, i)

                # Crossover
                trial = self.crossover_de(pop[i], mutant)

                # Evaluación
                trial_fitness = fitness_fn(trial)

                # Selección (reemplazo si es mejor)
                if trial_fitness > fitnesses[i]:
                    pop[i] = trial
                    fitnesses[i] = trial_fitness

                    # Actualizar mejor global
                    if trial_fitness > best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                        best_idx = i

            # Estadísticas
            avg_fit = np.mean(fitnesses)
            worst_fit = np.min(fitnesses)
            std_fit = np.std(fitnesses)

            history['best'].append(best_fitness)
            history['avg'].append(avg_fit)
            history['worst'].append(worst_fit)
            history['std'].append(std_fit)

            # Progreso
            progress = (g + 1) / self.generations * 100
            bar_len = 30
            filled = int(bar_len * (g + 1) / self.generations)
            bar = '█' * filled + '░' * (bar_len - filled)

            improvement = "🔥" if best_fitness == history['best'][-1] and g > 0 else "  "

            print(f"Gen {g+1:3d}/{self.generations} [{bar}] {progress:5.1f}% {improvement} | "
                  f"Mejor: {best_fitness:7.4f} | Prom: {avg_fit:7.4f} | "
                  f"Std: {std_fit:6.4f}")

        print(f"\n🔥 Evolución finalizada. Mejor fitness: {best_fitness:.4f}")

        return best_individual, best_fitness, history