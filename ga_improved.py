import numpy as np
import random

class ImprovedGeneticAlgorithm:
    def __init__(self, pop_size, generations, mutation_rate, crossover_rate,
                 n_params, elite_size=3, diversity_threshold=0.01):
        """
        GA Mejorado con múltiples operadores y estrategias adaptativas

        Nuevas características:
        - Múltiples tipos de crossover
        - Mutación direccional
        - Control de diversidad
        - Memoria de mejores soluciones
        """
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_params = n_params
        self.elite_size = elite_size
        self.diversity_threshold = diversity_threshold

        # Memoria de las mejores soluciones encontradas
        self.hall_of_fame = []
        self.hall_of_fame_size = 5

    def init_population(self):
        """Inicialización con LÍMITES para evitar grises"""
        pop = []

        for _ in range(self.pop_size):
            # LÍMITES ESTRICTOS para cada parámetro
            individual = np.array([
                np.random.uniform(0.35, 0.75),  # wrinkle: 35-75%
                np.random.uniform(0.20, 0.50),  # desaturation: MAX 50%
                np.random.uniform(0.20, 0.50),  # fade_color: MAX 50%
                np.random.uniform(0.35, 0.75)   # edge_strength: 35-75%
            ])
            pop.append(individual)

        return pop

    def select_tournament(self, pop, fitnesses, generation, total_generations):
        """Selección por torneo adaptativa"""
        # Aumentar presión de selección con el tiempo
        k = int(3 + 3 * (generation / total_generations))
        k = min(k, len(pop))

        participantes = random.sample(list(zip(pop, fitnesses)), k)
        participantes.sort(key=lambda x: x[1], reverse=True)
        return participantes[0][0]

    def crossover_arithmetic(self, p1, p2, alpha=0.5):
        """
        Crossover aritmético: child = alpha*p1 + (1-alpha)*p2
        Preserva mejor las buenas combinaciones
        """
        if random.random() < self.crossover_rate:
            alpha = random.uniform(0.3, 0.7)  # Alpha variable
            child = alpha * p1 + (1 - alpha) * p2
            return np.clip(child, 0, 1)
        return p1.copy()

    def crossover_blx_alpha(self, p1, p2, alpha=0.5):
        """
        BLX-α crossover: Explora un rango alrededor de los padres
        Mejor para exploración continua
        """
        if random.random() < self.crossover_rate:
            child = np.zeros(self.n_params)
            for i in range(self.n_params):
                cmin = min(p1[i], p2[i])
                cmax = max(p1[i], p2[i])
                range_val = cmax - cmin

                # Expandir rango con alpha
                lower = cmin - alpha * range_val
                upper = cmax + alpha * range_val

                child[i] = random.uniform(max(0, lower), min(1, upper))

            return child
        return p1.copy()

    def crossover_two_point(self, p1, p2):
        """Crossover de dos puntos"""
        if random.random() < self.crossover_rate:
            point1 = random.randint(0, self.n_params - 1)
            point2 = random.randint(point1, self.n_params)

            child = p1.copy()
            child[point1:point2] = p2[point1:point2]
            return child
        return p1.copy()

    def crossover_adaptive(self, p1, p2, generation, total_generations):
        """
        Crossover adaptativo: Cambia de estrategia según la generación
        - Inicio: BLX-α (exploración)
        - Medio: Aritmético (balance)
        - Final: Two-point (explotación)
        """
        progress = generation / total_generations

        if progress < 0.3:
            return self.crossover_blx_alpha(p1, p2, alpha=0.5)
        elif progress < 0.7:
            return self.crossover_arithmetic(p1, p2)
        else:
            return self.crossover_two_point(p1, p2)

    def mutate_directional(self, individual, best_individual, generation, total_generations):
        """Mutación con LÍMITES ESTRICTOS"""
        if random.random() < self.mutation_rate:
            sigma = 0.2 * np.exp(-4 * generation / total_generations)
            direction = best_individual - individual
            noise = np.random.normal(0, sigma, self.n_params)
            directional_factor = 0.3 * (1 - generation / total_generations)

            individual = individual + noise + directional_factor * direction

            # ⚠️ LÍMITES CRÍTICOS para evitar grises
            individual[0] = np.clip(individual[0], 0.35, 0.75)  # wrinkle
            individual[1] = np.clip(individual[1], 0.20, 0.50)  # desaturation MAX 50%
            individual[2] = np.clip(individual[2], 0.20, 0.50)  # fade_color MAX 50%
            individual[3] = np.clip(individual[3], 0.35, 0.75)  # edge_strength

        return individual

    def calculate_diversity(self, pop):
        """Calcula la diversidad de la población"""
        if len(pop) < 2:
            return 1.0

        # Diversidad como desviación estándar promedio
        pop_array = np.array(pop)
        diversity = np.mean(np.std(pop_array, axis=0))
        return diversity

    def inject_diversity(self, pop, fitnesses):
        """Inyecta diversidad si la población se estanca"""
        diversity = self.calculate_diversity(pop)

        if diversity < self.diversity_threshold:
            # Reemplazar 20% peores individuos con nuevos aleatorios
            num_replace = int(self.pop_size * 0.2)

            # Ordenar por fitness
            sorted_indices = np.argsort(fitnesses)

            # Reemplazar peores
            for i in range(num_replace):
                idx = sorted_indices[i]
                pop[idx] = np.random.rand(self.n_params)

            print(f"  💉 Diversidad baja ({diversity:.4f}), inyectando nuevos individuos")

        return pop

    def update_hall_of_fame(self, individual, fitness):
        """Mantiene las mejores soluciones históricas"""
        self.hall_of_fame.append((individual.copy(), fitness))
        self.hall_of_fame.sort(key=lambda x: x[1], reverse=True)
        self.hall_of_fame = self.hall_of_fame[:self.hall_of_fame_size]

    def run(self, fitness_fn):
        """Ejecuta el AG mejorado"""
        pop = self.init_population()
        history = {'best': [], 'avg': [], 'worst': [], 'std': [], 'diversity': []}

        # Evaluar población inicial
        fitnesses = [fitness_fn(ind) for ind in pop]
        best_ever = max(fitnesses)
        best_ever_individual = pop[np.argmax(fitnesses)].copy()

        for g in range(self.generations):
            # Evaluar fitness
            fitnesses = [fitness_fn(ind) for ind in pop]

            # Ordenar población
            ranked = sorted(list(zip(pop, fitnesses)), key=lambda x: x[1], reverse=True)

            # Actualizar mejor histórico
            if ranked[0][1] > best_ever:
                best_ever = ranked[0][1]
                best_ever_individual = ranked[0][0].copy()

            # Actualizar hall of fame
            self.update_hall_of_fame(ranked[0][0], ranked[0][1])

            # Élites (mejores de esta generación + hall of fame)
            elites = [p for p, f in ranked[:self.elite_size]]

            # Agregar uno del hall of fame si es mejor
            if len(self.hall_of_fame) > 0:
                if self.hall_of_fame[0][1] > ranked[self.elite_size-1][1]:
                    elites[-1] = self.hall_of_fame[0][0]

            # Estadísticas
            best_fit = ranked[0][1]
            worst_fit = ranked[-1][1]
            avg_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            diversity = self.calculate_diversity(pop)

            history['best'].append(best_fit)
            history['avg'].append(avg_fit)
            history['worst'].append(worst_fit)
            history['std'].append(std_fit)
            history['diversity'].append(diversity)

            # Barra de progreso mejorada
            progress = (g + 1) / self.generations * 100
            bar_len = 30
            filled = int(bar_len * (g + 1) / self.generations)
            bar = '█' * filled + '░' * (bar_len - filled)

            # Indicador de mejora
            improvement = "🔥" if best_fit == best_ever else "  "

            print(f"Gen {g+1:3d}/{self.generations} [{bar}] {progress:5.1f}% {improvement} | "
                  f"Mejor: {best_fit:7.4f} | Prom: {avg_fit:7.4f} | "
                  f"Div: {diversity:6.4f}")

            # Inyectar diversidad si es necesario
            pop = self.inject_diversity(pop, fitnesses)

            # Nueva generación
            new_pop = elites.copy()

            while len(new_pop) < self.pop_size:
                # Selección
                p1 = self.select_tournament(pop, fitnesses, g, self.generations)
                p2 = self.select_tournament(pop, fitnesses, g, self.generations)

                # Crossover adaptativo
                child = self.crossover_adaptive(p1, p2, g, self.generations)

                # Mutación direccional
                child = self.mutate_directional(child, best_ever_individual, g, self.generations)

                new_pop.append(child)

            pop = new_pop

        # Evaluar última generación
        fitnesses = [fitness_fn(ind) for ind in pop]
        best_idx = np.argmax(fitnesses)
        best_ind = pop[best_idx]
        best_fit = fitnesses[best_idx]

        # Usar el mejor histórico si es mejor
        if best_ever > best_fit:
            best_ind = best_ever_individual
            best_fit = best_ever

        print(f"\n🔥 Evolución finalizada. Mejor fitness obtenido: {best_fit:.4f}")
        print(f"📊 Hall of Fame (Top 5):")
        for i, (ind, fit) in enumerate(self.hall_of_fame[:5], 1):
            print(f"   #{i}: Fitness={fit:.4f}, Params={ind}")

        return best_ind, best_fit, history