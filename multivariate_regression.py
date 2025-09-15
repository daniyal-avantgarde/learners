import random

def predict(x, weights):
    return weights[0] * (weights[1] ** x)

def compute_loss(weights, data):
    return sum((y_true - predict(x, weights))**2 for x, y_true in data)

# Evolutionary parameters
population_size = 20
generations = 1000
mutation_strength = 0.1  # size of random mutations

# Initialize random population
population = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(population_size)]
data = [(1, 13.5914091),
        (2, 36.9452805),
        (3, 100.427685),
        (4, 272.99075)]

for gen in range(generations):
    # Evaluate loss for each individual
    scored_population = [(weights, compute_loss(weights, data)) for weights in population]
    scored_population.sort(key=lambda x: x[1])  # sort by loss (smaller is better)
    
    # Keep top 50% as parents
    parents = [w for w, l in scored_population[:population_size//2]]
    
    # Produce next generation by mutating parents
    population = []
    for parent in parents:
        # Keep parent
        population.append(parent)
        # Add mutated child
        child = [w + random.uniform(-mutation_strength, mutation_strength) for w in parent]
        population.append(child)
    
    best_weights = scored_population[0][0]
    best_loss = scored_population[0][1]
    print(f"Gen {gen+1}: Loss={best_loss:.4f}, Weights={best_weights}")
