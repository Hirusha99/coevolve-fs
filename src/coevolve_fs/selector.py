import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class CoevolutionarySelector:
    def __init__(self, classifier, population_size=100, generations=10, 
                 crossover_prob=0.8, mutation_prob=0.01):
        self.classifier = classifier
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.optimized_mask = None

    def divide_subgroups(self, df, columns_per_subgroup):
        """Divides dataframe columns into subgroups for co-evolution."""
        subgroups = [df.columns[i:i + columns_per_subgroup] for i in
                     range(0, len(df.columns), columns_per_subgroup)]
        return subgroups

    def _initialize_population(self, length):
        return np.random.randint(2, size=(self.population_size, length))

    def _evaluate_fitness(self, population, X_train, y_train, X_test, y_test):
        fitness = []
        for chromosome in population:
            # Handle edge case: No features selected
            if not np.any(chromosome):
                fitness.append(0.0)
                continue
            
            # Select features based on bitmask
            selected_cols = X_train.columns[chromosome == 1]
            
            self.classifier.fit(X_train[selected_cols], y_train)
            y_pred = self.classifier.predict(X_test[selected_cols])
            fitness.append(accuracy_score(y_test, y_pred))
        return np.array(fitness)

    def _selection(self, population, fitness):
        """Tournament Selection"""
        selected_parents = []
        for _ in range(2):
            idx = np.random.choice(len(population), size=3, replace=False)
            winner_idx = idx[np.argmax(fitness[idx])]
            selected_parents.append(population[winner_idx])
        return selected_parents

    def _crossover(self, p1, p2):
        if np.random.rand() < self.crossover_prob and len(p1) > 2:
            pts = np.sort(np.random.choice(range(1, len(p1)), 2, replace=False))
            c1 = np.concatenate((p1[:pts[0]], p2[pts[0]:pts[1]], p1[pts[1]:]))
            c2 = np.concatenate((p2[:pts[0]], p1[pts[0]:pts[1]], p2[pts[1]:]))
            return c1, c2
        return p1.copy(), p2.copy()

    def _mutate(self, child):
        mask = np.random.rand(len(child)) < self.mutation_prob
        child[mask] = 1 - child[mask]
        return child

    def fit(self, X_train, y_train, X_test, y_test, columns_per_subgroup=10):
        """Runs the co-evolutionary GA across feature subgroups."""
        subgroups = self.divide_subgroups(X_train, columns_per_subgroup)
        full_optimized_mask = []

        for i, subgroup in enumerate(subgroups):
            print(f"Evolving Subgroup {i+1}/{len(subgroups)}: {len(subgroup)} features")
            
            X_tr_sub = X_train[subgroup]
            X_te_sub = X_test[subgroup]
            
            population = self._initialize_population(len(subgroup))

            for gen in range(self.generations):
                fitness = self._evaluate_fitness(population, X_tr_sub, y_train, X_te_sub, y_test)
                new_pop = []
                
                while len(new_pop) < self.population_size:
                    p1, p2 = self._selection(population, fitness)
                    c1, c2 = self._crossover(p1, p2)
                    new_pop.extend([self._mutate(c1), self._mutate(c2)])
                
                population = np.array(new_pop)[:self.population_size]

            # Get best solution for this subgroup
            final_fit = self._evaluate_fitness(population, X_tr_sub, y_train, X_te_sub, y_test)
            best_sub_mask = population[np.argmax(final_fit)]
            full_optimized_mask.extend(best_sub_mask)

        self.optimized_mask = np.array(full_optimized_mask)
        return self.optimized_mask
