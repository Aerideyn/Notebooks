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

def compute_gradient(scalar, n_columns, dx, dy):
    grad_x = np.zeros(n_nodes)
    grad_y = np.zeros(n_nodes)
    for i in range(n_nodes):
        if i % n_columns != 0:  # not on the left boundary
            grad_p_x[i] = (scalar[i] - scalar[i-1]) / dx
        if i % n_columns != n_columns - 1:  # not on the right boundary
            grad_p_x[i] = (scalar[i+1] - scalar[i]) / dx
        if i >= n_columns:  # not on the top boundary
            grad_p_y[i] = (scalar[i] - scalar[i-n_columns]) / dy
        if i < n_nodes - n_columns:  # not on the bottom boundary
            grad_p_y[i] = (scalar[i+n_columns] - scalar[i]) / dy
    return grad_x, grad_y


n_rows    = 100
n_columns = 100
n_nodes   = n_rows*n_columns
mu        = 1.8e-5

# spacing between nodes
dx = 1
dy = 1
dz = 1

# _ _ _ _ _ _
#|_|_|_|_|_|_|
#|_|_|_|_|_|_|
#|_|_|_|_|_|_|


# the incompressible solver solves the momentum equation for the velocity field then uses the velocity field to solve the pressure field and then uses the pressure field to solve the velocity field again
# the incompressible solver is used for low mach number flows where the density is constant.

# initialize the velocity vector
u   = np.zeros(n_nodes)  # x component of velocity
v   = np.zeros(n_nodes)  # y component of velocity
p   = np.zeros(n_nodes)  # pressure field
rho = np.ones(n_nodes) # density field

# define the momentum equation coefficient matrix for each component of the velocity field
A_m_u = np.zeros((n_nodes, n_nodes))
A_m_v = np.zeros((n_nodes, n_nodes))

# calculate the pressure gradient in both directions
pressure_gradient = compute_gradient(p, n_columns, dx, dy)



# add the convection term to the momentum equation coefficient matrix
for i in range(n_nodes):
    if i % n_columns != 0:  # not on the left boundary
        aW = rho[i] * u[i] / dx
        A_m_u[i][i] += aW
    if i % n_columns != n_columns - 1:  # not on the right boundary
        A_m_u[i][i]   += -mu / dx**2
        A_m_u[i][i+1] += mu / dx**2
    if i >= n_columns:  # not on the top boundary
        A_m_v[i][i] += -mu / dy**2
        A_m_v[i][i-n_columns] += mu / dy**2
    if i < n_nodes - n_columns:  # not on the bottom boundary
        A_m_v[i][i] += -mu / dy**2
        A_m_v[i][i+n_columns] += mu / dy**2







    

    






