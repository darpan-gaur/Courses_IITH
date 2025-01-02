import numpy as np

def shape_fn(nen, zeta, le):
    """
    Shape functions for 2-noded elements.

    Parameters
    ----------
    nen : int
        Number of element nodes
    zeta : float
        Natural coordinate
    le : float
        Element length

    Returns
    -------
    N : ndarray
        Array of shape functions
    d2N : ndarray
        Array of second derivatives of shape functions

    By:
        Ishaan Jain
        CO21BTECH11006
    """
    if nen == 2:
        Nu1 = 0.25 * ((1-zeta)**2) * (2 + zeta)
        Nth1 = (le / 8) * ((1-zeta)**2) * (1+zeta)
        Nu2 = 0.25 * ((1+zeta)**2) * (2 - zeta)
        Nth2 = (le / 8) * ((1+zeta)**2) * (zeta-1)

        N = np.array([Nu1, Nth1, Nu2, Nth2])

        d2N = (1/le) * np.array([
            6*zeta/le,
             3*zeta - 1,
             -6*zeta/le,
             3*zeta + 1])

        return N, d2N
    else:
        raise NotImplementedError("Only 2-noded elements are supported")