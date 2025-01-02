import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_deflection(node_coordinates, displacements):
    """
    Plot the deflection of the beam.

    Parameters
    ----------
    node_coordinates : ndarray
        Array of nodal coordinates
    displacements : ndarray 
        Array of nodal displacements

    Saves the plot as 'deflection.png'

    By:
        Yoshita Kondapalli
        CO21BTECH11008
    """
    plt.figure(figsize=(8,6))

    plt.plot(node_coordinates, np.zeros_like(node_coordinates), label='Undeformed Beam', linestyle='--', color='gray')
    plt.plot(node_coordinates, displacements, label='Deformed Beam', color='blue')

    plt.title('Beam Deflection')
    plt.xlabel('Node Postition')
    plt.ylabel('Deflection')
    plt.legend()
    plt.grid(True)
    plt.savefig('deflection.png')

    print("Deflection plot saved as 'deflection.png'\n")

def plot_stress_profile(data, F):
    """
    Plot the stress profile along the length of the beam.

    Parameters
    ----------
    data : dict
        Dictionary containing input data
    F : ndarray
        Array of nodal forces   

    Saves the plot as 'bending_stress.png' and 'shear_stress.png'
    By:
        Yoshita Kondapalli
        CO21BTECH11008
    """

    # Stress profile
    # E = data['Material_Props']['Youngs_modulus']
    I = data['Material_Props']['Moment_of_inertia']
    L = data['Nodal_coords'][-1] -  data['Nodal_coords'][0]   # Length of the beam in meters
    y_max = 0.1  # Assuming half-height of the beam cross-section in meters

    # Node positions and corresponding bending moments (example values)
    node_positions = data['Nodal_coords']
    bending_moments = F[1::2].reshape(-1)
    shear_forces = F[::2].reshape(-1)

    # Interpolate the bending moment along the length of the beam
    moment_interpolation = interp1d(node_positions, bending_moments, kind='linear')
    shear_interpolation = interp1d(node_positions, shear_forces, kind='linear')

    # Discretize the length of the beam for plotting
    x_values = np.linspace(0, L, 100)  # 100 points along the length
    M_values = moment_interpolation(x_values)
    V_values = shear_interpolation(x_values)

    # Generate points across the cross-section height (y-axis, from -y_max to +y_max)
    y_values = np.linspace(-y_max, y_max, 50)  # 50 points across the cross-section

    # Initialize the stress array for storing stress values at each (x, y) point
    bending_stress = np.zeros((len(y_values), len(x_values)))
    shear_stress = np.zeros((len(y_values), len(x_values)))

    # Calculate stress at each point (x, y) using sigma_x = -My/I
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            h = 2*y_max
            bending_stress[j, i] = -M_values[i] * y / I  # Stress at (x, y)
            shear_stress[j, i] = V_values[i] * (h**2/4 - y**2) / (2 * I)

    # Create a contour plot for stress distribution
    X, Y = np.meshgrid(x_values, y_values)

    # Plot the bending stress distribution
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, bending_stress, levels=50, cmap='RdBu_r')
    plt.colorbar(contour, label='Bending Stress (MPa)')
    plt.title('Bending Stress Distribution Along the Beam')
    plt.xlabel('Length of the Beam (m)')
    plt.ylabel('Cross-Section Height (m)')
    plt.grid(True)
    plt.savefig('bending_stress.png')

    print("Bending stress plot saved as 'bending_stress.png'\n")

    # Plot the shear stress distribution
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, shear_stress, levels=50, cmap='RdBu_r')
    plt.colorbar(contour, label='Shear Stress (MPa)')
    plt.title('Shear Stress Distribution Along the Beam')
    plt.xlabel('Length of the Beam (m)')
    plt.ylabel('Cross-Section Height (m)')
    plt.grid(True)
    plt.savefig('shear_stress.png')

    print("Shear stress plot saved as 'shear_stress.png'\n")