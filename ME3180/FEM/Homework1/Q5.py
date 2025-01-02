# Name          : Darpan Gaur
# Roll Number   : CO21BTECH11004

import numpy as np

# Q5
# Inverse of matrix using adjoint method

def determinant(A):
    # calculate determinant custom
    if A.shape == (2,2):
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    det = 0
    for i in range(A.shape[0]):
        det += ((-1)**i)*A[0,i]*determinant(A[1:,np.arange(A.shape[1])!=i])
    return det

def inverse_matrix(A):
    # check if matrix is square
    if A.shape[0] != A.shape[1]:
        print("Matrix is not square")
        return None

    # calculate determinant
    det = determinant(A)
    if det == 0:
        print("Matrix is not invertible as det(Matrix) = 0")
        return None
    
    # calculate adjoint
    adj = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            adj[i,j] = ((-1)**(i+j))*determinant(A[np.arange(A.shape[0])!=i][:,np.arange(A.shape[1])!=j])

    # calculate inverse
    A_inv = adj.T/det
    return A_inv

# 3x3 matrix
A = np.array([[-1,3,-2],
              [2,-4,2],
              [0,4,1]])

print("Matrix A:")
print(A)

print("Inverse of A by adjoint method:")
print(inverse_matrix(A))

# Inverse of matrix by row reduction method

def row_swap(A,i,j):
    A[[i,j]] = A[[j,i]]
    return A

def row_addition(A,i,j,k):
    A[j] = A[j] - k*A[i]
    return A

def row_multiplication_scaler(A,i,k):
    A[i] = k*A[i]
    return A

def row_reduction(A):
    # check if matrix is square
    if A.shape[0] != A.shape[1]:
        print("Matrix is not square")
        return None

    # check if matrix is invertible
    if determinant(A) == 0:
        print("Matrix is not invertible as det(Matrix) = 0")
        return None
    
    # do for zero diagonal elements
    for i in range(A.shape[0]):
        if A[i,i] == 0:
            for j in range(i+1,A.shape[0]):
                if A[j,i] != 0:
                    A = row_swap(A,i,j)
                    break

    # augment matrix with identity matrix
    A = np.hstack((A,np.eye(A.shape[0])))

    # perform row operations
    for i in range(A.shape[0]):
        A = row_multiplication_scaler(A,i,1/A[i,i])
        for j in range(A.shape[0]):
            if i != j:
                A = row_addition(A,i,j,A[j,i])

    # extract inverse matrix
    A_inv = A[:,A.shape[0]:]
    return A_inv

# 3x3 matrix
A = np.array([[-1,3,-2],
              [2,-4,2],
              [0,4,1]])

print("Matrix A:")
print(A)

print("Inverse of A by row reduction method:")
print(row_reduction(A))
