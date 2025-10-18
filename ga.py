import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, pop_size, generations, mutation_rate, crossover_rate, n_params):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_params = n_params

    # ----------------------------
    # Inicialización de población
    # ----------------------------
    def init_population(self):
        return [np.random.rand(self.n_params) for _ in range(self.pop_size)]

    # ----------------------------
    # Selección por torneo (k=3)
    # ----------------------------
    def select_tournament(self, pop, fitnesses, k=3):
        participantes = random.sample(list(zip(pop, fitnesses)), k)
        participantes.sort(key=lambda x: x[1], reverse=True)
        return participantes[0][0]  # devuelve el mejor del torneo

    # ----------------------------
    # Crossover uniforme
    # ----------------------------
    def crossover_uniform(self, p1, p2, prob=0.5):
        if random.random() < self.crossover_rate:
            mask = np.random.rand(len(p1)) < prob
            child = np.where(mask, p1, p2)
            return child
        return p1.copy()

    # ----------------------------
    # Mutación gaussiana adaptativa
    # ----------------------------
    def mutate(self, individual, generation, total_generations):
        sigma = 0.1 * (1 - generation / total_generations)  # disminuye con el tiempo
        if random.random() < self.mutation_rate:
            individual += np.random.normal(0, sigma, self.n_params)
            individual = np.clip(individual, 0, 1)
        return individual

    # ----------------------------
    # Ciclo principal del algoritmo
    # ----------------------------
    def run(self, fitness_fn):
        pop = self.init_population()

        for g in range(self.generations):
            fitnesses = [fitness_fn(ind) for ind in pop]

            # Ranking global (de mayor a menor)
            ranked = sorted(list(zip(pop, fitnesses)), key=lambda x: x[1], reverse=True)
            elites = [p for p, f in ranked[:2]]  # elitismo: mantener 2 mejores

            # Información de progreso
            best_fit = ranked[0][1]
            avg_fit = np.mean(fitnesses)
            print(f"Generación {g+1}/{self.generations} | Mejor: {best_fit:.4f} | Promedio: {avg_fit:.4f}")

            # Nueva población
            new_pop = elites.copy()

            while len(new_pop) < self.pop_size:
                # Selección
                p1 = self.select_tournament(pop, fitnesses, k=3)
                p2 = self.select_tournament(pop, fitnesses, k=3)

                # Crossover
                child = self.crossover_uniform(p1, p2, prob=0.5)

                # Mutación adaptativa
                child = self.mutate(child, g, self.generations)

                new_pop.append(child)

            pop = new_pop

        # Evaluar última generación
        fitnesses = [fitness_fn(ind) for ind in pop]
        best_idx = np.argmax(fitnesses)
        best_ind = pop[best_idx]
        best_fit = fitnesses[best_idx]

        print(f"\n🏁 Evolución finalizada. Mejor fitness obtenido: {best_fit:.4f}")
        return best_ind, best_fit
