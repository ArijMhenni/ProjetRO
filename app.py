from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import numpy as np

# Sample data: Distance matrix
n = 5  # Number of cities
np.random.seed(32)
dist_matrix = np.random.randint(10, 100, size=(n, n))
np.fill_diagonal(dist_matrix, 0)

# Create Gurobi model
model = Model("TSP")

# Decision variables
x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")

# Objective function
model.setObjective(quicksum(dist_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

# Constraints
# 1. Each city is departed once
for i in range(n):
    model.addConstr(quicksum(x[i, j] for j in range(n) if i != j) == 1)

# 2. Each city is entered once
for j in range(n):
    model.addConstr(quicksum(x[i, j] for i in range(n) if i != j) == 1)

# 3. Subtour elimination constraints (MTZ)
for i in range(1, n):
    for j in range(1, n):
        if i != j:
            model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

# Optimize
model.optimize()

# Extract solution
if model.status == GRB.OPTIMAL:
    solution = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if x[i, j].x > 0.5:
                solution[i, j] = 1

    print("Optimal tour:")
    print(solution)

    # Extract the tour sequence
    tour = []
    current_city = 0
    while len(tour) < n:
        tour.append(current_city)
        next_city = np.where(solution[current_city] == 1)[0][0]
        current_city = next_city
    tour.append(0)  # Return to the start city
    print("Tour:", tour)

    # Plotting
    plt.figure(figsize=(8, 6))
    x_coords = np.random.rand(n)
    y_coords = np.random.rand(n)
    for i in range(n):
        plt.scatter(x_coords[i], y_coords[i], c="red")
        plt.text(x_coords[i], y_coords[i], str(i), fontsize=12, ha="right")
    for i in range(len(tour) - 1):
        plt.plot([x_coords[tour[i]], x_coords[tour[i + 1]]], 
                 [y_coords[tour[i]], y_coords[tour[i + 1]]], "b-")
    plt.title("Optimal TSP Tour")
    plt.show()
else:
    print("No optimal solution found.")
