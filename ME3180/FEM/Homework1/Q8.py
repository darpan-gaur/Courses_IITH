# Name          : Darpan Gaur
# Roll Number   : CO21BTECH11004

import numpy as np

def custum_linagSolve(A, B):
    '''
    Solve linear equation Ax = B
    '''
    n = A.shape[0]
    x = np.zeros([n, 1])
    for i in range(n):
        x[i] = (B[i] - np.sum(A[i, np.arange(n)!=i]*x[np.arange(n)!=i]))/A[i,i]
    return x

def oneSpringStiffness(k):
    '''
    Return 2x2 stiffness matrix for element
    '''
    K = k*np.array([
        [1, -1],
        [-1, 1]
    ])
    return K

def computeStiffnessSystem(Spring_nodes ,K , Ke):
    '''
    Assemble the global stiffness matrix
    '''
    if (Spring_nodes.ndim == 1):
        idx = Spring_nodes - 1
        K[np.ix_(idx, idx)] += Ke
    else:
        for row in Spring_nodes:
            idx = row - 1
            K[np.ix_(idx, idx)] += Ke
    return K

# get global force and displacement
def get_force_disp(K, f, bc_zero = None, bc_cust = None):
    '''
    Solve the system of equations to get global reaction force and displacement
    '''
    n_nodes = K.shape[0]
    n_nodesBC = bc_zero.shape[0]

    if bc_cust is None:
        bc_cust = np.zeros([n_nodesBC])

    if bc_zero is None:
        return np.linalg.solve(K, f)
    
    bc_idx = np.ones(n_nodes, 'bool')
    bc_disp = np.arange(n_nodes)

    bc_idx[np.ix_(bc_zero-1)] = False
    bc_disp = bc_disp[bc_idx]

    fSys = f[bc_disp] - K[np.ix_((bc_disp), (bc_zero-1))] * np.asmatrix(bc_cust.reshape(n_nodesBC, 1))
    # uSys = np.linalg.solve(K[np.ix_((bc_disp), (bc_disp))], fSys)
    uSys = custum_linagSolve(K[np.ix_(bc_disp, bc_disp)], fSys)

    u = np.zeros([n_nodes, 1])
    u[np.ix_(bc_zero-1)] = np.asmatrix(bc_cust.reshape(n_nodesBC, 1))
    u[np.ix_(bc_disp)] = uSys

    Fsys = K*np.asmatrix(u) - f

    return np.asmatrix(u), Fsys

# Inputs

# setting nodes and elements
E_n1_n2 = np.array([
    [1, 3],     # spring between node 1 and 3
    [2, 3],     # spring between node 2 and 3
    [3, 4]      # spring between node 3 and 4
])

# setting spring stiffness
k = np.array([500.0, 500.0, 1000.0])

# setting stiffness matrix
K = np.zeros([4, 4])

# setting force
# f = np.array([0.0, 0.0, 4.0, 0.0])
f = np.zeros([4, 1])
f[2] = 4.0

# setting boundary conditions
bc = np.array([1, 2, 4])    # nodes 1, 2 and 4 are fixed

# get stiffness matrix for each spring
Ke = np.array([oneSpringStiffness(ki) for ki in k])

# assemble the global stiffness matrix
computeStiffnessSystem(E_n1_n2[0, :], K, Ke[0])
computeStiffnessSystem(E_n1_n2[1, :], K, Ke[1])
computeStiffnessSystem(E_n1_n2[2, :], K, Ke[2])

# print the global stiffness matrix
print('Global Stiffness Matrix:')
print(K)

# solve
u, F = get_force_disp(K, f, bc_zero = bc)
# u, F = get_disp_reac(K, f, bc_zero = bc)

# print the displacement and reaction force
print('Displacement:')
print(u)

print('Reaction Force:')
print(F + f)