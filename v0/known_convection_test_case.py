import numpy as np


def gauss_seidel(A, b, x0, epsilon, max_iterations):
    n = len(A)
    x = x0.copy()
    for i in range(max_iterations):
        print(i)
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

def gmres(A, b, x0, epsilon, max_iterations):
    n = len(A)
    x = x0.copy()
    r = b - np.dot(A, x)
    p = r.copy()
    for i in range(max_iterations):
        print(i)
        Ap = np.dot(A, p)
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r) < epsilon:
            print("converged after: " +str(i) + " iterations")
            return x
        beta = np.dot(r, r) / np.dot(r - alpha * Ap, r - alpha * Ap)
        p = r + beta * p
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

# assemble the advection matrix using the upwind discretization scheme
def assemble_advection_matrix_uds(n_rows, n_columns, dx, dy, dz, rho, u, v):
    coefficient_matrix = np.zeros([n_nodes, n_nodes])
    for row in range(n_rows):
        for column in range(n_columns):
            node_id_C = get_node_id(row, column)
            
            # if the node is not on the boundary
            if row > 0 and row < n_rows - 1 and column > 0 and column < n_columns - 1:
                # get the node ids
                node_id_E = get_node_id(row, column + 1)
                node_id_S = get_node_id(row - 1, column)
                node_id_W = get_node_id(row, column - 1)
                node_id_N = get_node_id(row + 1, column)
                
                # calculate the mass flow rates and dot them with the face normal vectors
                mdot_e =  1.0 * rho * dy * dz * max(u[node_id_E], 0.0)
                mdot_s = -1.0 * rho * dx * dz * max(v[node_id_S], 0.0)
                mdot_w = -1.0 * rho * dy * dz * min(u[node_id_W], 0.0)
                mdot_n =  1.0 * rho * dx * dz * min(v[node_id_N], 0.0)
                
                # if the total mass flow rate is not zero
                if abs(mdot_e + mdot_s + mdot_w + mdot_n) > 1e-10:
                    print("Warning: mass flow rates do not sum to zero at node: " + str(node_id_C))

                # add the coefficients to the matrix
                coefficient_matrix[node_id_C][node_id_E] += -mdot_e
                coefficient_matrix[node_id_C][node_id_S] += -mdot_s
                coefficient_matrix[node_id_C][node_id_W] += -mdot_w
                coefficient_matrix[node_id_C][node_id_N] += -mdot_n
                coefficient_matrix[node_id_C][node_id_C] += mdot_e + mdot_s + mdot_w + mdot_n

    return coefficient_matrix

def assemble_advection_matrix_cds(n_rows, n_columns, dx, dy, dz, rho, u ,v):
    coefficient_matrix = np.zeros([n_nodes, n_nodes])
    for row in range(n_rows):
        for column in range(n_columns):
            # if the node is not on the boundary
            if row > 0 and row < n_rows - 1 and column > 0 and column < n_columns - 1:
                # get the node ids
                node_id_C = get_node_id(row, column)
                node_id_E = get_node_id(row, column + 1)
                node_id_S = get_node_id(row - 1, column)
                node_id_W = get_node_id(row, column - 1)
                node_id_N = get_node_id(row + 1, column)
                
                # calculate the mass flow rates and dot them with the face normal vectors
                mdot_e =  1.0 * rho * dy * dz * (u[node_id_E] + u[node_id_C]) * 0.5
                mdot_s = -1.0 * rho * dx * dz * (v[node_id_S] + v[node_id_C]) * 0.5
                mdot_w = -1.0 * rho * dy * dz * (u[node_id_W] + u[node_id_C]) * 0.5
                mdot_n =  1.0 * rho * dx * dz * (v[node_id_N] + v[node_id_C]) * 0.5
                
                # if the total mass flow rate is not zero
                if abs(mdot_e + mdot_s + mdot_w + mdot_n) > 1e-10:
                    print("Warning: mass flow rates do not sum to zero at node: " + str(node_id_C))

                # add the coefficients to the matrix
                coefficient_matrix[node_id_C][node_id_E] += -0.5 * mdot_e
                coefficient_matrix[node_id_C][node_id_S] += -0.5 * mdot_s
                coefficient_matrix[node_id_C][node_id_W] += -0.5 * mdot_w
                coefficient_matrix[node_id_C][node_id_N] += -0.5 * mdot_n
                coefficient_matrix[node_id_C][node_id_C] += -0.5 * (mdot_n + mdot_s + mdot_w + mdot_e)

    return coefficient_matrix

def assemble_diffusion_matrix(n_rows, n_columns, dx, dy, dz, diffusion_coefficient):    
    # the coefficient matrix
    coefficient_matrix = np.zeros([n_nodes, n_nodes])
    for row in range(n_rows):
        for column in range(n_columns):
            node_id_C = get_node_id(row, column)
            
            # insert east neighbor into A
            if column < n_columns -1:
                a1 = diffusion_coefficient * (dy*dz) / dx
                coefficient_matrix[node_id_C][get_node_id(row, column + 1)] = a1
                coefficient_matrix[node_id_C][node_id_C] -= a1

            #insert south neighbor into A
            if row > 0:
                a2 = diffusion_coefficient * (dx*dz) / dy
                coefficient_matrix[node_id_C][get_node_id(row - 1, column)] = a2
                coefficient_matrix[node_id_C][node_id_C] -= a2

            #insert west neighbor into A
            if column > 0:
                a3 = diffusion_coefficient * (dy*dz) / dx
                coefficient_matrix[node_id_C][get_node_id(row, column - 1)] = a3
                coefficient_matrix[node_id_C][node_id_C] -= a3

            # insert north neighbor into A
            if row < n_rows -1:
                a4 = diffusion_coefficient * (dx*dz) / dy
                coefficient_matrix[node_id_C][get_node_id(row + 1, column)] = a4
                coefficient_matrix[node_id_C][node_id_C] -= a4

    return coefficient_matrix

def apply_diffusion_dirichlet_bc(coefficient_matrix, source_vector, node_id, diffusion_coefficient, value):
    source_vector[node_id] -= diffusion_coefficient * value
    coefficient_matrix[node_id][node_id] -= diffusion_coefficient

def apply_neumann_bc(source_vector, node_id, flux):
    source_vector[node_id] += flux

def solve_linear_system(coeff_matrix, constant_vector, initial_guess, tolerance, max_iterations):
    preconditioner = np.diag(1 / np.diag(coeff_matrix))
    preconditioned_matrix = np.dot(preconditioner, coeff_matrix)
    preconditioned_vector = np.dot(preconditioner, constant_vector)
    solution = conjugate_gradient(preconditioned_matrix, preconditioned_vector, initial_guess, tolerance, max_iterations)
    return solution

n_rows        = 80
n_columns     = 80
n_nodes       = n_rows*n_columns
conductivity  = 0.009 #alu w/m/k
rho           = 1.0   # air kg/m3
starting_temp = 0.5

# spacing between nodes
dx = 1.0 / (n_columns)
dy = 1.0 / (n_rows)
dz = 1.0

# node grid
x = np.linspace(0.5 * dx, n_columns * dx - 0.5 * dx, n_columns)
y = np.linspace(0.5 * dy,    n_rows * dy - 0.5 * dy, n_rows)

# the variable vector
solution_vector = np.full(n_nodes, starting_temp)

# the constant vector (source terms)
source_vector = np.full((n_nodes), 0.0)

uu, vv = np.meshgrid(x, -y)
u = uu.flatten()
v = vv.flatten()

diffusion_matrix   = assemble_diffusion_matrix(n_rows, n_columns, dx, dy, dz, conductivity)
convection_matrix  = assemble_advection_matrix_cds(n_rows, n_columns, dx, dy, dz, rho, u, v)
coefficient_matrix = diffusion_matrix + convection_matrix

# apply 0 phi along the top boundary
for column in range(n_columns):
    k = conductivity * (dx*dz) / (dy / 2.0)
    apply_diffusion_dirichlet_bc(coefficient_matrix, source_vector, get_node_id(-1, column), k, 0.0)


# apply a variable phi along the left boundary
phi_left = np.linspace(1.0, 0.0, n_rows)
for row in range(n_rows):
    k = conductivity * (dy*dz) / (dx / 2.0)
    apply_diffusion_dirichlet_bc(coefficient_matrix, source_vector, get_node_id(row, 0), k, phi_left[row])
    
# apply 

#T = solve_linear_system(coefficient_matrix, source_vector, solution_vector, 1e-5, 10000)
#T = conjugate_gradient(coefficient_matrix, source_vector, solution_vector, 1e-5, 100000)
#T = gauss_seidel(coefficient_matrix, source_vector, solution_vector, 1e-5, 10000)
T = gmres(coefficient_matrix, source_vector, solution_vector, 1e-5, 1000)

import matplotlib.pyplot as plt


t_reshaped = np.reshape(T, (-1, n_columns))

#plt.figure()
#plt.streamplot(x, y, u.reshape(n_rows, n_columns), v.reshape(n_rows, n_columns))
#plt.grid()

plt.figure()
levels = np.arange(0.05, .96, 0.1)
plt.contour(t_reshaped, levels=levels)
plt.colorbar()
plt.grid()

#plt.figure()
#plt.plot(t_reshaped[5])
#plt.grid()
#
#plt.figure()
#plt.plot(t_reshaped[:,0])
#plt.grid()
#
#plt.figure()
#plt.contour(np.reshape(u*v, (-1, n_columns)), levels=30)
#plt.grid()


plt.show()