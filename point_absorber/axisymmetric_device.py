import os
import logging
import numpy as np
import capytaine as cpt
import matplotlib.pyplot as plt
from scipy.special import comb

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:\t%(message)s")

# Problem setup
draft = 5
show_mesh = False
plot_results = False
device_mass = 1e6  # TODO: use shape profile to generate mass and stiffness values
device_stiffness = 1e7

os.system('cls')


# Set up shape profile r(z) for the axisymmetric body
def shape(z):
    return 0.1 * (-(z + 1) ** 2 + 16)


###############################################################
###############################################################

# Trying this method for generating the Bezier Curve:
# https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier_curve(points, ntimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    # yPoints = np.array([p[1] for p in points])
    zPoints = np.array([p[2] for p in points])

    t = np.linspace(0.0, 1.0, ntimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    # yvals = np.dot(yPoints, polynomial_array)
    zvals = np.dot(zPoints, polynomial_array)
    profile_pts = np.asarray([[xvals[i], 0, zvals[i]] for i in range(ntimes)])

    return xvals, zvals, profile_pts


def make_mesh(points_array, excursion=0):

    # Close the bottom of the points
    bottom = np.array(points_array[0])
    bottom_spacing = np.linspace(0, bottom[0], 6)
    bottom_pts = np.asarray([[bottom_spacing[i], 0, bottom[2]] for i in range(5)])
    mesh_shape = np.concatenate((bottom_pts, points_array), axis=0)

    # Make mesh and add DOF
    buoy = cpt.FloatingBody(cpt.AxialSymmetricMesh.from_profile(profile=mesh_shape, nphi=40))
    buoy.add_translation_dof(name="Heave")

    # Calculate the mass and stiffness parameters
    volume = buoy.mesh.volume
    mass = volume * 1000
    r = points_array[-1][0]
    stiffness = np.pi * 1000 * 9.81 * r**2 * (1 - (excursion**2 / (3 * r**2)))

    return mesh_shape, buoy, mass, stiffness


if show_mesh:
    buoy.show()

# Set up radiation and diffraction problems
omega_range = np.linspace(0.3, 5.0, 60)
problems = [cpt.RadiationProblem(body=buoy, radiating_dof='Heave', omega=omega)
            for omega in omega_range]
problems += [cpt.DiffractionProblem(omega=omega, body=buoy, wave_direction=0.0)
             for omega in omega_range]

# Solve the problems using the axial symmetry
solver = cpt.BEMSolver(engine=cpt.HierarchicalToeplitzMatrixEngine())  # TODO: investigate why this engine is uses
results = [solver.solve(pb) for pb in sorted(problems)]
*radiation_results, diffraction_result = results
dataset = cpt.assemble_dataset(results)

# Assemble needed values for control algorithm
added_mass = dataset['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave')
radiation_damping = dataset['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave')
dataset['excitation_force'] = dataset['Froude_Krylov_force'] + dataset['diffraction_force']
excitation_force = dataset['excitation_force'].sel(wave_direction=0.0)  # is this already the fourier transform?

# Control algorithm  # TODO: get this working : )
intrinsic_impedance = radiation_damping + 1.0j * omega_range * \
                      (device_mass + added_mass - device_stiffness / (omega_range ** 2))
power_take_off_impedance = np.conjugate(intrinsic_impedance)
optimal_velocity = excitation_force / (2 * np.real(intrinsic_impedance))
power_take_off_force = -1.0 * power_take_off_impedance * optimal_velocity
optimal_power = 0.5 * power_take_off_force * optimal_velocity

# Plot results
if plot_results:
    plt.figure()
    plt.plot(omega_range, optimal_power, marker='o')
    plt.xlabel('omega')
    plt.ylabel('Optimal Power')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Create z and x points
    zpts = np.linspace(-5, 0, 10)
    xpts = np.random.uniform(0, 5, size=(10, 1))
    xyz_pts = np.asarray([[xpts[i][0], 0, zpts[i]] for i in range(10)])
    bez_x, bez_y, bez_pts = bezier_curve(xyz_pts, ntimes=100)

    buoy_shape, buoy_mesh, buoy_mass, buoy_stiffness = make_mesh(bez_pts)
    buoy_mesh.show()
