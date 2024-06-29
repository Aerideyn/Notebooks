import numpy as np


# _ _ _ _ _ _
#|_|_|_|_|_|_|
#|_|_|_|_|_|_|
#|_|_|_|_|_|_|

# number of rows and columns and nodes
n_rows        = 80
n_columns     = 80
n_nodes       = n_rows*n_columns

# spacing between nodes
dx = 1.0 / (n_columns)
dy = 1.0 / (n_rows)
dz = 1.0

# define the diffusion coefficient
diffusion_coefficient = 237

# define the source vector and the diffusion matrix
source_vector = np.zeros(n_nodes)
diffusion_matrix = np.zeros([n_nodes, n_nodes])

# define the node x and y coordinates  
node_x   = np.linspace(0.5 * dx, n_columns * dx - 0.5 * dx, n_columns)
node_y   = np.linspace(0.5 * dy,    n_rows * dy - 0.5 * dy, n_rows)
node_u   = np.zeros(n_nodes)  # x component of velocity
node_v   = np.zeros(n_nodes)  # y component of velocity
node_phi = np.zeros(n_nodes)  # scalar field
node_rho = np.ones(n_nodes)   # density field

# define the face connectivity and normal arrays
int_face_to_elem = []
bdy_face_to_elem = []
elem_to_int_face = [[] for i in range(n_nodes)]
elem_to_bdy_face = [[] for i in range(n_nodes)]

int_face_area      = []
int_face_norm      = []

bdy_face_norm  = []
bdy_face_area  = []
bdy_face_type  = [] # 0=neumann, 1=dirichlet
bdy_face_value = [] 

# define the functions.
def get_node_id(row, column):
    return row*n_columns + column

def id_to_row(id):
    return id // n_columns

def id_to_column(id):
    return id % n_columns

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

# initialize the face connectivity and normal arrays for both internal and boundary faces
for row in range(n_rows):
    for column in range(n_columns):
        # if the node is not on the boundary:
        if 0 < row < n_rows - 1 and 0 < column < n_columns - 1:
            int_face_to_elem.append([get_node_id(row, column), get_node_id(row + 1, column)]) # north face
            int_face_to_elem.append([get_node_id(row, column), get_node_id(row, column + 1)]) # east face
            int_face_to_elem.append([get_node_id(row, column), get_node_id(row - 1, column)]) # south face
            int_face_to_elem.append([get_node_id(row, column), get_node_id(row, column - 1)]) # west face

            int_face_norm.append(np.array([0, 1])) # north face
            int_face_norm.append(np.array([1, 0])) # east face
            int_face_norm.append(np.array([0, -1])) # south face
            int_face_norm.append(np.array([-1, 0])) # west face

            int_face_area.append(dx * dz) # north face
            int_face_area.append(dy * dz) # east face
            int_face_area.append(dx * dz) # south face
            int_face_area.append(dy * dz) # west face

        elif row == 0:
            bdy_face_to_elem.append([get_node_id(row, column)]) # south boundary face
            bdy_face_area.append(dx * dz) # south face
            bdy_face_norm.append(np.array([0, -1]))
            bdy_face_type.append(0) # neumann boundary condition
            bdy_face_value.append(0) # zero flux

        elif row == n_rows - 1:
            bdy_face_to_elem.append([get_node_id(row, column)]) # north boundary face
            bdy_face_norm.append(np.array([0, 1]))
            bdy_face_area.append(dx * dz)
            bdy_face_type.append(0) # neumann boundary condition
            bdy_face_value.append(0) # zero flux

        elif column == 0:
            bdy_face_to_elem.append([get_node_id(row, column)]) # west boundary face
            bdy_face_norm.append(np.array([-1, 0]))
            bdy_face_area.append(dy * dz)
            bdy_face_type.append(1) # neumann boundary condition
            bdy_face_value.append(300) # zero flux

        elif column == n_columns - 1:
            bdy_face_to_elem.append([get_node_id(row, column)]) # east boundary face
            bdy_face_norm.append(np.array([1, 0]))
            bdy_face_area.append(dy * dz)
            bdy_face_type.append(0) # neumann boundary condition
            bdy_face_value.append(1000) # zero flux

# initialize the element to face connectivity
for face_index in range(len(int_face_to_elem)):
    elem_to_int_face[int_face_to_elem[face_index][0]].append(face_index)
    elem_to_int_face[int_face_to_elem[face_index][1]].append(face_index)

# initialize the element to face connectivity for the boundary faces
for face_index in range(len(bdy_face_to_elem)):
    elem_to_bdy_face[bdy_face_to_elem[face_index][0]].append(face_index)
        

# loop through the internal faces and add to the coefficient matrix
for face_index in range(len(int_face_to_elem)):
    node_id_1   = int_face_to_elem[face_index][0]
    node_id_2   = int_face_to_elem[face_index][1]
    
    #compute the distance between node1 and node2
    node1_row = id_to_row(node_id_1)
    node1_column = id_to_column(node_id_1)
    node2_row = id_to_row(node_id_2)
    node2_column = id_to_column(node_id_2)

    distance = np.sqrt((node_x[node1_column] - node_x[node2_column])**2 + (node_y[node1_row] - node_y[node2_row])**2)
    diffusion_coefficient_face = diffusion_coefficient * int_face_area[face_index] / distance

    diffusion_matrix[node_id_1][node_id_1] += diffusion_coefficient_face
    diffusion_matrix[node_id_1][node_id_2] -= diffusion_coefficient_face
    diffusion_matrix[node_id_2][node_id_2] += diffusion_coefficient_face
    diffusion_matrix[node_id_2][node_id_1] -= diffusion_coefficient_face

# loop through the boundary faces and apply the boundary conditions
for face_index in range(len(bdy_face_to_elem)):
    node_id = bdy_face_to_elem[face_index][0]
    if bdy_face_type[face_index] == 0: # neumann boundary condition
        source_vector[node_id] += bdy_face_value[face_index] * bdy_face_area[face_index]

    if bdy_face_type[face_index] == 1: # dirichlet boundary condition
        a = diffusion_coefficient * bdy_face_area[face_index] / (dx / 2.0)
        diffusion_matrix[node_id][node_id] -= a
        source_vector[node_id] -= a * bdy_face_value[face_index]

# test calculation to verify the result.
gl = diffusion_coefficient * (n_rows*dy*dz) / (n_columns*dx) # W/K
power = 1000 #w/m2
goal_T = 300 + (power * (n_rows*dy*dz))/gl
print(gl)
print(goal_T)

# solve the linear system
solution = gmres(diffusion_matrix, source_vector, np.zeros(n_nodes), 1e-6, 10)
#solution = gauss_seidel(diffusion_matrix, source_vector, np.zeros(n_nodes), 1e-6, 1000)

# plot the solution
import matplotlib.pyplot as plt
plt.contourf(solution.reshape(n_rows, n_columns))
plt.colorbar()
plt.grid()
plt.show()


