import numpy as np


def gauss_seidel(A, b, x0, epsilon, max_iterations):
    n = len(A)
    x = x0.copy()
    for i in range(max_iterations):
        x_new = np.zeros(n)
        for j in range(n):
            s1 = np.dot(A[j, :j], x_new[:j])
            s2 = np.dot(A[j, j + 1:], x[j + 1:])
            x_new[j] = (b[j] - s1 - s2) / A[j, j]

        if np.allclose(x, x_new, rtol=epsilon):
            print("converged after: " +str(i) + " iterations")
            return x_new
        x = x_new
    return x

def conjugate_gradient(coeff_matrix, constant_vector, initial_guess, tolerance, max_iterations):
    x = initial_guess.copy()
    residual = constant_vector - np.dot(coeff_matrix, x)
    direction = residual.copy()
    residual_squared = np.dot(residual, residual)

    for i in range(max_iterations):
        print(i)
        Ap = np.dot(coeff_matrix, direction)
        alpha = residual_squared / np.dot(direction, Ap)
        x = x + alpha * direction
        residual = residual - alpha * Ap
        new_residual_squared = np.dot(residual, residual)
        if new_residual_squared < tolerance * tolerance:
            print("Converged after " + str(i) + " iterations")
            return x
        direction = residual + (new_residual_squared / residual_squared) * direction
        residual_squared = new_residual_squared

    return x

def get_node_id(row, column):
    return row*n_columns + column

def assemble_diffusion_matrix(n_rows, n_columns, dx, dy, dz, diffusion_coefficient):    
    # the coefficient matrix
    coefficient_matrix = np.zeros([n_nodes, n_nodes])
    for row in range(n_rows):
        for column in range(n_columns):
            node_id_C = row*n_columns + column
            
            # insert east neighbor into A
            if column < n_columns -1:
                a1 = -diffusion_coefficient * (dy*dz) / dx
                coefficient_matrix[node_id_C][get_node_id(row, column + 1)] = a1
                coefficient_matrix[node_id_C][node_id_C] -= a1

            #insert north neighbor into A
            if row > 0:
                a2 = -diffusion_coefficient * (dx*dz) / dy
                coefficient_matrix[node_id_C][get_node_id(row - 1, column)] = a2
                coefficient_matrix[node_id_C][node_id_C] -= a2

            #insert west neighbor into A
            if column > 0:
                a3 = -diffusion_coefficient * (dy*dz) / dx
                coefficient_matrix[node_id_C][get_node_id(row, column - 1)] = a3
                coefficient_matrix[node_id_C][node_id_C] -= a3

            # insert south neighbor into A
            if row < n_rows -1:
                a4 = -diffusion_coefficient * (dx*dz) / dy
                coefficient_matrix[node_id_C][get_node_id(row + 1, column)] = a4
                coefficient_matrix[node_id_C][node_id_C] -= a4

    return coefficient_matrix

def assemble_advection_matrix(n_rows, n_columns, dx, dy, dz, rho, u ,v):
    # the coefficient matrix
    coefficient_matrix = np.zeros([n_nodes, n_nodes])
    for row in range(n_rows):
        for column in range(n_columns):
            node_id_C = get_node_id(row, column)
            
            # insert east neighbor into A
            if column < n_columns -1:
                mdot = rho * (u[node_id_C] + u[get_node_id(row, column + 1)]) / 2
                a1 = mdot * (dy*dz) / 2
                coefficient_matrix[node_id_C][get_node_id(row, column + 1)] = a1
                coefficient_matrix[node_id_C][node_id_C] += a1

            #insert north neighbor into A
            if row > 0:
                mdot = rho * (v[node_id_C] + v[get_node_id(row - 1, column)]) / 2
                a2 = mdot * (dx*dz) / 2
                coefficient_matrix[node_id_C][get_node_id(row - 1, column)] = a2
                coefficient_matrix[node_id_C][node_id_C] += a2

            #insert west neighbor into A
            if column > 0:
                mdot = rho * (u[node_id_C] + u[get_node_id(row, column - 1)]) / 2
                a3 = -mdot * (dy*dz) / 2
                coefficient_matrix[node_id_C][get_node_id(row, column - 1)] = a3
                coefficient_matrix[node_id_C][node_id_C] += a3

            # insert south neighbor into A
            if row < n_rows -1:
                mdot = rho * (v[node_id_C] + v[get_node_id(row + 1, column)]) / 2
                a4 = -mdot * (dx*dz) / 2
                coefficient_matrix[node_id_C][get_node_id(row + 1, column)] = a4
                coefficient_matrix[node_id_C][node_id_C] += a4

    return coefficient_matrix

def apply_dirichlet_boundary_conditions(coefficient_matrix, source_vector, node_id, value):
    coefficient_matrix[node_id][:] = 0
    coefficient_matrix[node_id][node_id] = 1
    source_vector[node_id] = value

def apply_neumann_boundary_conditions(source_vector, node_id, value):
    source_vector[node_id] = value

def solve_linear_system(coeff_matrix, constant_vector, initial_guess, tolerance, max_iterations):
    preconditioner = np.diag(1 / np.diag(coeff_matrix))
    preconditioned_matrix = np.dot(preconditioner, coeff_matrix)
    preconditioned_vector = np.dot(preconditioner, constant_vector)
    solution = conjugate_gradient(preconditioned_matrix, preconditioned_vector, initial_guess, tolerance, max_iterations)
    return solution


n_rows        = 20
n_columns     = 20
n_nodes       = n_rows*n_columns
conductivity  = 237 #alu w/m/k
rho           = 1   # air kg/m3
starting_temp = 0

# spacing between nodes
dx = 1 #/ n_columns
dy = 1 #/ n_rows
dz = 1

# test calculation to verify the result.
gl = conductivity * (n_rows*dy*dz) / (n_columns*dx) # W/K
power = 1000 #w/m2
goal_T = 300 + (power * (n_rows*dy*dz))/gl
print(gl)
print(goal_T)

# the variable vector
solution_vector = np.full(n_nodes, starting_temp)

# the constant vector (source terms)
source_vector = np.full((n_nodes), 0)


u = np.linspace(0,  1, n_nodes)
v = np.linspace(0, -1, n_nodes)

coefficient_matrix = assemble_diffusion_matrix(n_rows, n_columns, dx, dy, dz, conductivity)
#coefficient_matrix = np.add(coefficient_matrix, assemble_advection_matrix(n_rows, n_columns, dx, dy, dz, rho, u, v))

# apply boundary conditions
for row in range(n_rows):
    apply_dirichlet_boundary_conditions(coefficient_matrix, source_vector, get_node_id(row, 0), 300)
    apply_neumann_boundary_conditions(source_vector, get_node_id(row, n_columns - 1), 1000 * (dx*dz))


T = solve_linear_system(coefficient_matrix, source_vector, solution_vector, 1e-6, 10000)
#T = conjugate_gradient(coefficient_matrix, source_vector, solution_vector, 1e-6, 10000)
#T = gauss_seidel(coefficient_matrix, source_vector, solution_vector, 1e-6, 10000)

import matplotlib.pyplot as plt

t_reshaped = np.reshape(T, (-1, n_columns))

plt.contourf(t_reshaped)
plt.colorbar()
plt.grid()

plt.figure()
plt.plot(t_reshaped[5])
plt.grid()

plt.show()