# Name        :- Darpan Gaur
# Roll Number :- CO21BTECH11004

# import libraries
import numpy as np
from numpy.polynomial.legendre import leggauss

# define functions
def q3_a(x):
    return x**2 + 1

def q3_b(x):
    return x**4 + 2*x**2

def q3_c(x):
    return x/(x**2 + 1)

def q3_d(x):
    return np.cos(np.pi*x)**2

def q3_e(x):
    return 3*x**3 + 2

def gauss_quad_integral(func, a, b, n):
    # Get the Gauss-Legendre points and weights
    points, weights = leggauss(n)
    
    # Perform change of interval [a, b] -> [-1, 1]
    transformed_points = 0.5 * (points + 1) * (b - a) + a
    transformed_weights = 0.5 * (b - a) * weights
    
    # Compute the integral using the transformed points and weights
    integral = np.sum(transformed_weights * func(transformed_points))
    
    return integral

# Gauss quadrature integration for all cases
r = 2
print(f"q3_a: using {r} points")
print("Gauss quadrature integral: ", gauss_quad_integral(q3_a, 0, 4, r))

r = 3
print(f"q3_b: using {r} points")
print("Gauss quadrature integral: ", gauss_quad_integral(q3_b, -1, 1, r))

r = 2
print(f"q3_c: using {r} points")
print("Gauss quadrature integral: ", gauss_quad_integral(q3_c, -1, 1, r))

r = 5
print(f"q3_d: using {r} points")
print("Gauss quadrature integral: ", gauss_quad_integral(q3_d, -1, 1, r))

r = 2
print(f"q3_e: using {r} points")
print("Gauss quadrature integral: ", gauss_quad_integral(q3_e, -1, 1, r))