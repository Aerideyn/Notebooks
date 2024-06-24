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

n_rows        = 100
n_columns     = 100
n_nodes       = n_rows*n_columns
conductivity  = 237 #alu w/m/k
starting_temp = 0

# spacing between nodes
dx = 1
dy = 1
dz = 1

# _ _ _ _ _ _
#|_|_|_|_|_|_|
#|_|_|_|_|_|_|
#|_|_|_|_|_|_|

gl = conductivity * (n_rows*dy*dz) / (n_columns*dx) # W/K
power = 1000 #w/m2
goal_T = 300 + (power * (n_rows*dy*dz))/gl
print(gl)
print(goal_T)

# the coefficient matrix
node_gamma = np.zeros([n_nodes, n_nodes])

# the variable vector
node_phi   = np.full((n_nodes), starting_temp)

# the constant vector (source terms)
node_b     = np.full((n_nodes), 0)

for row in range(n_rows):
    for column in range(n_columns):
        node_id_C = row*n_columns + column
        ac = 0

        # insert right neighbor into A
        if column < n_columns -1:
            node_id_R = node_id_C + 1
            a1 = -conductivity * (dy*dz) / dx #GL to right neighbor
            node_gamma[node_id_C][node_id_R] = a1
            ac += a1

        #insert up neighbor into A
        if row > 0:
            a2 = -conductivity * (dx*dz) / dy #GL to up neighbor
            node_id_U = node_id_C - n_columns
            node_gamma[node_id_C][node_id_U] = a2
            ac += a2

        #insert left neighbor into A
        if column > 0:
            a3 = -conductivity * (dy*dz) / dx #GL to left neighbor
            node_id_L = node_id_C - 1
            node_gamma[node_id_C][node_id_L] = a3
            ac += a3

        # insert down neighbor into A
        if row < n_rows -1:
            a4 = -conductivity * (dx*dz) / dy #GL to down neighbor
            node_id_D = node_id_C + n_columns 
            node_gamma[node_id_C][node_id_D] = a4
            ac += a4

        ### handle boundaries ###
        if column == 0:
            ab = -conductivity * (dy*dz) / (dx / 2.0)
            node_b[node_id_C] = -ab * 300.0 # dirilecht condition for 300 kelvin.
            ac += ab

        ### handle boundaries ###
        if column == n_columns - 1:
            node_b[node_id_C] = 1000 * (dx*dz) # von neumann condition for 1000W /m2

        #insert this cell into A
        node_gamma[node_id_C][node_id_C] = -ac




T_cg = conjugate_gradient(node_gamma, node_b, node_phi, 1e-6, 10000)
T = T_cg

import matplotlib.pyplot as plt

t_reshaped = np.reshape(T, (-1, n_columns))

plt.contourf(t_reshaped)
plt.colorbar()
plt.grid()

plt.figure()
plt.plot(t_reshaped[5])
plt.grid()

plt.show()