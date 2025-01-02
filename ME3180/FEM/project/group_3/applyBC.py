import numpy as np

def apply_bc(K_global, F_global, prescribed_dofs):
  """
  Modify global stiffness matrix and force vector based on
  prescribed boundary conditions.

  Parameters
  ----------
  K_global : ndarray
      Global stiffness matrix
  F_global : ndarray
      Global force vector
  prescribed_dofs : list
      List of prescribed degrees of freedom
      (node, dof, value) format

  Returns
  -------
  K_t : ndarray
      Modified global stiffness matrix
  F_t : ndarray
      Modified global force vector

  By: 
    Gayathri Shreeya
    CO21BTECH11010
  """
  K_t = K_global.copy()
  F_t = F_global.copy()
  for dof in prescribed_dofs:
    node, dir, val = map(int, dof[:3])
    temp = (node - 1) * 2 + (dir - 1)
    K_t[temp, :] = 0.0
    K_t[:, temp] = 0.0
    K_t[temp, temp] = 1.0
    F_t[temp] = val
  return K_t, F_t