import logging
import numpy as np
import capytaine as cpt
import matplotlib.pyplot as plt
from scipy.special import comb

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:\t%(message)s")

# Generating the Bezier Curve:
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
        ntimes is the number of time steps, defaults to 1000

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


def make_mesh(points_array):

    # Close the bottom of the points
    bottom = np.array(points_array[0])
    bottom_spacing = np.linspace(0, bottom[0], 6)
    bottom_pts = np.asarray([[bottom_spacing[i], 0, bottom[2]] for i in range(5)])
    mesh_shape = np.concatenate((bottom_pts, points_array), axis=0)

    # Make mesh and add the vertical degree of freedom
    buoy = cpt.FloatingBody(cpt.AxialSymmetricMesh.from_profile(profile=mesh_shape, nphi=40))
    buoy.add_translation_dof(name="Heave")

    # Calculate the mass and stiffness parameters
    volume = buoy.mesh.volume
    mass = volume * 1000
    r = points_array[-1][0]
    stiffness = np.pi * 1000 * 9.81 * r**2

    return mesh_shape, buoy, mass, stiffness

def evaluate_buoy_forces(buoy_object):
    # Set up radiation and diffraction problems
    problems = [cpt.RadiationProblem(body=buoy_object, radiating_dof='Heave', omega=omega)
                for omega in omega_range]
    problems += [cpt.DiffractionProblem(omega=omega, body=buoy_object, wave_direction=0.0)
                for omega in omega_range]

    # Solve each matrix problem
    solver = cpt.BEMSolver(engine=cpt.HierarchicalToeplitzMatrixEngine())  # TODO: investigate why this engine is used
    results = [solver.solve(pb) for pb in sorted(problems)]

    return cpt.assemble_dataset(results)

def complex_conjugate_control(device_data, device_mass, device_stiffness):
    # Assemble needed values for control algorithm
    added_mass = device_data['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave')
    radiation_damping = device_data['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave')
    device_data['excitation_force'] = device_data['Froude_Krylov_force'] + device_data['diffraction_force']
    excitation_force = device_data['excitation_force'].sel(wave_direction=0.0)  # is this already the fourier transform?

    # Control algorithm # TODO: add in extra viscous linear damping if needed
    intrinsic_impedance = radiation_damping + 1.0j * omega_range * \
                        (device_mass + added_mass - device_stiffness / (omega_range ** 2))
    power_take_off_impedance = np.conjugate(intrinsic_impedance)
    optimal_velocity = excitation_force / (2 * np.real(intrinsic_impedance))
    power_take_off_force = -1.0 * power_take_off_impedance * optimal_velocity
    optimal_power = 0.5 * power_take_off_force * optimal_velocity
    
    return optimal_power


def objective_function(bezier_radii):
    annual_power = 0.0

    return annual_power


if __name__ == '__main__':

    # User inputs
    omega_range = np.linspace(0.3, 5.0, 60)
    draft = 5
    radial_control_points = 8
    plot_results = False
    show_mesh = False

    # Create z and x points
    zpts = np.linspace(-draft, 0, radial_control_points)
    xpts = np.random.uniform(0, 5, size=(radial_control_points, 1))
    xyz_pts = np.asarray([[xpts[i][0], 0, zpts[i]] for i in range(radial_control_points)])
    bez_x, bez_y, bez_pts = bezier_curve(xyz_pts, ntimes=100)

    buoy_shape, buoy_mesh, buoy_mass, buoy_stiffness = make_mesh(bez_pts)

    if show_mesh:
        buoy_mesh.show()

    buoy_hydrodynamic_data = evaluate_buoy_forces(buoy_mesh)
    power_data = complex_conjugate_control(buoy_hydrodynamic_data, buoy_mass, buoy_stiffness)

    # Plot results
    if plot_results:
        plt.figure()
        plt.plot(omega_range, power_data, marker='o')
        plt.xlabel('omega')
        plt.ylabel('Optimal Power')
        plt.tight_layout()
        plt.show()
        