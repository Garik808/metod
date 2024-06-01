import numpy as np

# Параметры задачи
supply = np.array([50, 60, 25])
demand = np.array([40, 50, 45])
costs = np.array([
    [5, 8, 6],
    [9, 12, 7],
    [3, 4, 2]
])

# Проверка сбалансированности задачи
assert supply.sum() == demand.sum(), "Суммарное предложение должно равняться суммарному спросу."

# Параметры PSO
num_particles = 50
num_iterations = 200
w = 0.5  # Инерционное вес
c1 = 1.5  # Коэффициент когнитивной составляющей
c2 = 1.5  # Коэффициент социальной составляющей

# Генерация начальных частиц
particles = np.random.rand(num_particles, 3, 3)
velocities = np.zeros((num_particles, 3, 3))
personal_best_positions = particles.copy()
personal_best_costs = np.full(num_particles, np.inf)
global_best_position = None
global_best_cost = np.inf

def calculate_cost(matrix):
    return np.sum(matrix * costs)

def check_constraints(matrix):
    return np.all(matrix.sum(axis=1) <= supply) and np.all(matrix.sum(axis=0) <= demand)

for i in range(num_iterations):
    for j in range(num_particles):
        particles[j] = np.maximum(0, np.minimum(particles[j], supply[:, None]))
        particles[j] = np.maximum(0, np.minimum(particles[j].T, demand[:, None])).T
        
        if check_constraints(particles[j]):
            cost = calculate_cost(particles[j])
            if cost < personal_best_costs[j]:
                personal_best_costs[j] = cost
                personal_best_positions[j] = particles[j].copy()
                
            if cost < global_best_cost:
                global_best_cost = cost
                global_best_position = particles[j].copy()
    
    for j in range(num_particles):
        r1 = np.random.rand(3, 3)
        r2 = np.random.rand(3, 3)
        velocities[j] = (w * velocities[j] +
                         c1 * r1 * (personal_best_positions[j] - particles[j]) +
                         c2 * r2 * (global_best_position - particles[j]))
        particles[j] += velocities[j]

best_distribution = global_best_position
print("Лучшее распределение груза:")
print(best_distribution)
print("Минимальные затраты:")
print(global_best_cost)