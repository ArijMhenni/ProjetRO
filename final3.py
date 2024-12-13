from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_distance_matrix_from_csv(file_path):
    """Load a symmetric distance matrix from a CSV file with city names as headers."""
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

            # First row contains city names
            city_names = rows[0][1:]  # Skip the first empty cell

            # The rest is the distance matrix
            dist_matrix = []
            for row in rows[1:]:
                dist_matrix.append([float(value) for value in row[1:]])  # Skip the city name in the first column

            # Check for symmetry
            n = len(dist_matrix)
            for i in range(n):
                for j in range(n):
                    if dist_matrix[i][j] != dist_matrix[j][i]:
                        raise ValueError("Distance matrix is not symmetric.")

            return city_names, dist_matrix
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None

def fill_symmetric_matrix(n, city_names):
    """Prompt the user to fill a symmetric distance matrix."""
    dist_matrix = [[0] * n for _ in range(n)] # juste intialisation 

    print("Enter the distances between cities (symmetric matrix):")
    for i in range(n):
        for j in range(i + 1, n):
            while True:
                try:
                    dist = float(input(f"Distance from {city_names[i]} to {city_names[j]}: "))
                    if dist < 0:
                        raise ValueError("Distance cannot be negative.")
                    dist_matrix[i][j] = dist
                    dist_matrix[j][i] = dist
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please try again.")

    return dist_matrix

def tsp_solver(city_names, dist_matrix):
    n = len(city_names)

    # Create Gurobi model
    model = Model("TSP")

    # Decision variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    u = model.addVars(n, vtype=GRB.CONTINUOUS, name="u")

    # Objective function
    model.setObjective(quicksum(dist_matrix[i][j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Constraints
    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(n) if i != j) == 1)
        model.addConstr(quicksum(x[j, i] for j in range(n) if i != j) == 1)

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

        # Extract the tour sequence
        tour = []
        current_city = 0
        while len(tour) < n:
            tour.append(current_city)
            next_city = np.where(solution[current_city] == 1)[0][0]
            current_city = next_city
        tour.append(0)  # Return to the start city

        return tour, model.objVal
    else:
        return None, None

def plot_tsp(city_names, dist_matrix, tour):
    n = len(city_names)
    np.random.seed(70)
    x_coords = np.random.rand(n)
    y_coords = np.random.rand(n)

    plt.figure(figsize=(8, 6))
    for i in range(n):
        plt.scatter(x_coords[i], y_coords[i], c="red")
        plt.text(x_coords[i], y_coords[i], city_names[i], fontsize=12, ha="right")
    for i in range(len(tour) - 1):
        plt.plot([x_coords[tour[i]], x_coords[tour[i + 1]]], 
                 [y_coords[tour[i]], y_coords[tour[i + 1]]], "b-", label=f"{dist_matrix[tour[i]][tour[i+1]]:.2f}")
    plt.title("Optimal TSP Tour")
    plt.show()

def main():
    # Ask user whether to load a CSV file or enter manually
    print("How would you like to input the distance matrix?")
    print("1. Load from CSV file")
    print("2. Enter manually")

    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # Load distance matrix
    if choice == "1":
        file_path = input("Enter the path to the CSV file: ").strip()
        city_names, dist_matrix = load_distance_matrix_from_csv(file_path)
        if dist_matrix is None or city_names is None:
            print("Failed to load the distance matrix from the CSV file.")
            return

    else:
        # Enter number of cities
        while True:
            try:
                n = int(input("Enter the number of cities: "))
                if n <= 1:
                    raise ValueError("Number of cities must be greater than 1.")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")

        # Enter city names
        city_names = []
        for i in range(n):
            while True:
                name = input(f"Enter the name of city {i + 1}: ").strip()
                if name:
                    city_names.append(name)
                    break
                else:
                    print("City name cannot be empty. Please try again.")

        # Fill the distance matrix manually
        dist_matrix = fill_symmetric_matrix(n, city_names)

    # Solve TSP
    tour, cost = tsp_solver(city_names, dist_matrix)
    if tour is None:
        print("No optimal solution found.")
    else:
        tour_names = [city_names[i] for i in tour]
        print(f"Optimal tour: {' -> '.join(tour_names)}")
        print(f"Total cost: {cost:.2f}")
        plot_tsp(city_names, dist_matrix, tour)

if __name__ == "__main__":
    main()
