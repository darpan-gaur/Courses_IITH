import numpy as np
from shape_function import shape_fn
import matplotlib.pyplot as plt

def get_fel(data, id_a, id_b, nen):
    """
    Compute the element force vector for a given element.

    Parameters
    ----------
    data : dict
        Dictionary containing input data
    id_a : int
        Node ID of the first node of the element
    id_b : int
        Node ID of the second node of the element
    nen : int
        Number of element nodes

    Returns
    -------
    fel : ndarray
        Element force vector

    By:
        Darpan Gaur
        CO21BTECH11004
    """
    ngp = nen
    zeta, W = np.polynomial.legendre.leggauss(ngp)

    fel = np.zeros((2*nen, 1))
    xae = data['Nodal_coords'][id_a]
    xbe = data['Nodal_coords'][id_b]
    bodyforces = data['Body_forces']
    for q in bodyforces:
        coords = q['start_end_coords']
        x1, x2 = coords
        f1, f2 = q['val']
        fn = lambda x: (f1 + (f2 - f1) * (x - x1) / (x2 - x1))

        # Check if [xae, xbe] overlaps with the body force coordinates
        if xae > x2 or xbe < x1:
            continue
        else:
        # Find the overlapping region
            x_start = max(xae, x1)
            x_end = min(xbe, x2)

            le = x_end - x_start

            zeta_to_x = lambda zeta: ((x_end + x_start)/2 + ((x_end - x_start)/2)*zeta)

            # If le is close to 0, we can ignore the body force
            if abs(le) < 1e-10:
                continue

            for j in range(2*nen):
                val = 0.0
                for z, w in zip(zeta, W):
                    temp = shape_fn(nen, z, le)
                    Ni = temp[0][j]
                    val += Ni * fn(zeta_to_x(z)) * w
                val *= (le/2)
                fel[j] += val
    return fel

def get_kel(data, id_a, id_b, nen):
    """
    Compute the element stiffness matrix for a given element.

    Parameters
    ----------
    data : dict
        Dictionary containing input data
    id_a : int
        Node ID of the first node of the element
    id_b : int
        Node ID of the second node of the element
    nen : int
        Number of element nodes

    Returns
    -------
    kel : ndarray
        Element stiffness matrix

    By:
        Aaryan
        CO21BTECH11001
    """
    ngp = nen
    zeta, W = np.polynomial.legendre.leggauss(ngp)

    kel = np.zeros((2*nen, 2*nen))
    xae = data['Nodal_coords'][id_a]
    xbe = data['Nodal_coords'][id_b]
    E = data['Material_Props']['Youngs_modulus']
    I = data['Material_Props']['Moment_of_inertia']

    le = xbe - xae
    for i in range(2*nen):
        for j in range(2*nen):
            val = 0.0
            for z, w in zip(zeta, W):
                temp = shape_fn(nen, z, le)
                d2Ni = temp[1][i] # Value of d2N_i / dx2 at zeta
                d2Nj = temp[1][j] # Value of d2N_j / dx2 at zeta
                val += d2Ni * d2Nj * w
            val *= (E*I*(le/2))
            kel[i, j] = val
    return kel

def global_stiffness_force(data):
    """
    Assemble the global stiffness matrix and force vector.

    Parameters
    ----------
    data : dict
        Dictionary containing input data

    Returns
    -------
    K : ndarray
        Global stiffness matrix
    F : ndarray
        Global force vector

    By:
        Darpan Gaur
        CO21BTECH11004
    """
    nnodes = data["No._nodes"]
    nen = len(data['Element_connectivity'][0])

    K = np.zeros((2*nnodes, 2*nnodes))
    F = np.zeros((2*nnodes, 1))

    for i in range(len(data['Element_connectivity'])):
        elem = data['Element_connectivity'][i]
        elem = [x-1 for x in elem]  # Convert to 0-based indexing
    
        kel = get_kel(data, elem[0], elem[1], nen)
        fel = get_fel(data, elem[0], elem[1], nen)
    
        # Assemble
        for j in range(nen):
            for k in range(nen):
                K[2*elem[j], 2*elem[k]] += kel[2*j, 2*k]
                K[2*elem[j], 2*elem[k]+1] += kel[2*j, 2*k+1]
                K[2*elem[j]+1, 2*elem[k]] += kel[2*j+1, 2*k]
                K[2*elem[j]+1, 2*elem[k]+1] += kel[2*j+1, 2*k+1]

        # Assemble F
        for j in range(nen):
            F[2*elem[j]] += fel[2*j]
            F[2*elem[j]+1] += fel[2*j+1]

    for sp in data['Spring_stiffness']:
        node, stiffness = sp
        row_number = 2*(int(node)-1)
        K[row_number, row_number] += stiffness
    
    return K,F