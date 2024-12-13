from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox


def fill_symmetric_matrix(n, city_names):
    """Prompt the user to fill a symmetric distance matrix."""
    dist_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist = simpledialog.askfloat(
                "Input",
                f"Enter distance from {city_names[i]} to {city_names[j]}:",
            )
            if dist is None or dist < 0:
                messagebox.showerror("Error", "Invalid distance.")
                return None
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

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
    np.random.seed(42)
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
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Get number of cities
    n = simpledialog.askinteger("Input", "Enter the number of cities:")
    if n is None or n <= 1:
        messagebox.showerror("Error", "Invalid number of cities.")
        return

    # Get city names
    city_names = []
    for i in range(n):
        name = simpledialog.askstring("Input", f"Enter the name of city {i + 1}:")
        if not name:
            messagebox.showerror("Error", "City name cannot be empty.")
            return
        city_names.append(name)

    # Fill the distance matrix
    dist_matrix = fill_symmetric_matrix(n, city_names)
    if dist_matrix is None:
        return

    # Solve TSP
    tour, cost = tsp_solver(city_names, dist_matrix)
    if tour is None:
        messagebox.showinfo("Result", "No optimal solution found.")
    else:
        tour_names = [city_names[i] for i in tour]
        messagebox.showinfo("Result", f"Optimal tour: {' -> '.join(tour_names)}\nTotal cost: {cost:.2f}")
        plot_tsp(city_names, dist_matrix, tour)

if __name__ == "__main__":
    main()