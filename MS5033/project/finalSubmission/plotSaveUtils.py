from dolfin import *
import random
import numpy as np
import time
from mshr import *  # For complex meshes
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import path
from shutil import copyfile
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

def plotResult(cnt, resPath, param, row, col, figSaveName, plotType, cntLevels):
    fig, ax = plt.subplots(row, col, figsize=(20, 15))
    idx = 1
    for i in range(row):
        for j in range(col):
        # Load the .vtu file
        # idx = i*18 + j*7 + 1
            # if (idx)<10:
            #     # mesh = pv.read(f"/home/darpan/Desktop/Desktop/8thSem/MS5033/project/data2D__N=64/mu00000{idx}.vtu")
            #     mesh = pv.read(f"{resPath}/{param}00000{idx}.vtu")
            # else:
            #     # mesh = pv.read(f"/home/darpan/Desktop/Desktop/8thSem/MS5033/project/data2D__N=64/mu0000{idx}.vtu")
            #     mesh = pv.read(f"{resPath}/{param}0000{idx}.vtu")
            idx_str = str(idx).zfill(6)
            mesh = pv.read(f"{resPath}/{param}{idx_str}.vtu")
        # get the mesh points
            X = mesh.points[:, 0]
            Y = mesh.points[:, 1]
            val = mesh.active_scalars

            points_Q1 = np.stack([X, Y], axis=-1)
            values_Q1 = val

        # Reflect to 2nd quadrant (x -> -x, y)
            points_Q2 = np.stack([-X, Y], axis=-1)
            values_Q2 = values_Q1  # Same values due to symmetry

        # Reflect to 4th quadrant (x, y -> -y)
            points_Q4 = np.stack([X, -Y], axis=-1)
            values_Q4 = values_Q1  # Same values

        # Reflect to 3rd quadrant (x -> -x, y -> -y)
            points_Q3 = np.stack([-X, -Y], axis=-1)
            values_Q3 = values_Q1  # Same values

        # Combine all points and values
            points_full = np.concatenate([points_Q1, points_Q2, points_Q3, points_Q4], axis=0)
            values_full = np.concatenate([values_Q1, values_Q2, values_Q3, values_Q4], axis=0)

            if plotType == 'contour':
                sc = ax[i][j].tricontourf(points_full[:, 0], points_full[:, 1], values_full, levels=cntLevels, cmap='rainbow')
            elif plotType == 'scatter':
                sc = ax[i][j].scatter(points_full[:, 0], points_full[:, 1], c=values_full, cmap='rainbow', s=3)
            else:
                raise ValueError("Invalid plot type. Use 'scatter' or 'contour'.")
            tit = f't = {idx}'
            ax[i, j].set_title(tit)
            ax[i, j].set_xlabel('X-axis')
            ax[i, j].set_ylabel('Y-axis')
            plt.colorbar(sc, ax=ax[i][j], label='Value')
        # ax[i][j].axis('off')
            ax[i][j].set_xlim(-6, 6)
            ax[i][j].set_ylim(-6, 6)

            idx += cnt

# add title for big figure

    fig.suptitle(f'{param} visualization with time', fontsize=16)
    plt.tight_layout()
    plt.savefig(figSaveName, dpi=300)
    plt.show()

def configure_newton_solver():
    """Configure the Newton solver for the nonlinear problem"""
    solver = NewtonSolver()
    solver.parameters['relative_tolerance'] = 1e-6
    solver.parameters['linear_solver'] = 'lu'
    solver.parameters['krylov_solver']['absolute_tolerance'] = 1E-7
    solver.parameters['krylov_solver']['relative_tolerance'] = 1E-4
    solver.parameters['krylov_solver']['maximum_iterations'] = 1000
    
    return solver

def setup_form_compiler():
    """Configure form compiler options"""
    parameters["ghost_mode"] = "shared_facet"
    parameters["form_compiler"]["representation"] = "quadrature"
    parameters["form_compiler"]["quadrature_rule"] = "vertex"
    parameters["form_compiler"]["quadrature_degree"] = 1    # degree of quadrature

def makeAnimation(resPath, param):
    # write function to make animation
    t = 0

    # complete
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    ims = []

    for idx in range(0, 51, 10):
        mesh = pv.read(f"{resPath}/{param}0000{idx}.vtu")
        X = mesh.points[:, 0]
        Y = mesh.points[:, 1]
        val = mesh.active_scalars

        points_full = np.stack([X, Y], axis=-1)
        sc = ax.tricontourf(points_full[:, 0], points_full[:, 1], val, levels=50, cmap='rainbow')
        ims.append([sc])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    ani.save('animation.gif', writer='imagemagick')

def open_output_files(output_params, tumor_model):
    """Open files for VTU output"""
    file0 = None
    file1 = None
    file2 = None
    
    if not output_params.save_as_vtu:
        return (file0, file1, file2)
    
    file0 = File(output_params.output_path + "/phi.pvd", "compressed")
    
    if tumor_model.num_equations >= 2:
        file1 = File(output_params.output_path + "/mu.pvd", "compressed")
    
    if tumor_model.num_equations >= 3:
        file2 = File(output_params.output_path + "/sigma.pvd", "compressed")
    
    return (file0, file1, file2)


def save_vtu_output(output_params, tumor_model, sim_params, u, j, t, file0, file1=None, file2=None):
    """Save output for ParaView visualization"""
    if output_params.save_as_vtu and (j%output_params.output_freq==0 or j==sim_params.num_time_steps):
        file0 << (u.split()[0], t)
        
        if file1 is not None and tumor_model.num_equations >= 2:
            file1 << (u.split()[1], t)
        
        if file2 is not None and tumor_model.num_equations >= 3:
            file2 << (u.split()[2], t)