import logging
import numpy as np
import capytaine as cpt
import matplotlib.pyplot as plt
from scipy.special import comb
import random
import xarray as xr

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
    solver = cpt.BEMSolver(engine=cpt.HierarchicalToeplitzMatrixEngine())
    results = [solver.solve(pb) for pb in sorted(problems)]

    return cpt.assemble_dataset(results)

def complex_conjugate_control(device_data, device_mass, device_stiffness):
    # Assemble needed values for control algorithm
    added_mass = device_data['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave').data
    radiation_damping = device_data['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave').data
    friction_damping = 0.10 * np.max(radiation_damping)
    # TODO: adjust frictional damping if necessary

    # Assemble unit wave amplitude excitation force in same format as added mass and radiation damping arrays
    device_data['excitation_force'] = device_data['Froude_Krylov_force'] + device_data['diffraction_force']
    excitation_force = device_data['excitation_force'].sel(wave_direction=0.0).data  # is this already the fourier transform?
    excitation_force = excitation_force.transpose()
    excitation_force = excitation_force[0, :]

    # Calculate the random wave offset and excitation for each frequency component
    wave_spectra = bretschneider_mitsuyasu_spectrum(omega_range, significant_wave_height=3.5, significant_wave_period=11.0)
    d_omega = omega_range[1] - omega_range[0]
    random_phase_offset = 2 * np.pi * np.random.rand(len(omega_range))
    frequency_domain_wave_spectrum_amplitudes = 2 * np.sqrt(wave_spectra * d_omega) * np.exp(1.0j * random_phase_offset)
    excitation_force = excitation_force * frequency_domain_wave_spectrum_amplitudes

    # Complex conjugate control algorithm 
    intrinsic_impedance = (radiation_damping + friction_damping) + 1.0j * omega_range * \
                        (device_mass + added_mass - device_stiffness / (omega_range ** 2))
    power_take_off_impedance = np.conjugate(intrinsic_impedance)
    optimal_velocity = excitation_force / (2 * np.real(intrinsic_impedance))
    power_take_off_force = power_take_off_impedance * optimal_velocity
    optimal_power = 0.5 * power_take_off_force * np.conjugate(optimal_velocity)
    
    if plot_results:
        plt.plot(omega_range, added_mass, marker='o', label='Added Mass')
        plt.plot(omega_range, radiation_damping, marker='o', label='Added Mass')
        plt.plot(omega_range, np.abs(excitation_force), marker='o', label='Excitation Force')
        # TODO: check abs on excitation force magnitude

        plt.xlabel('$\omega$')
        plt.tight_layout()
        plt.grid()
        plt.legend()
        plt.show()


    return optimal_power

def bretschneider_mitsuyasu_spectrum(wave_radial_frequency_array, significant_wave_height, significant_wave_period):
    f = wave_radial_frequency_array / (2 * np.pi)

    h = significant_wave_height
    t = significant_wave_period

    spectra = 0.257 * (h ** 2) * (t ** -4) * (f ** -5) * np.exp(-1.03 * (t * f) ** -4)

    return spectra


def objective_function(power_data):
    annual_power = -1.0 * np.sum(np.real(power_data))
    print(annual_power)

    return annual_power


if __name__ == '__main__':

    # User inputs
    omega_range = np.linspace(0.3, 2.0, 40)
    draft = 5
    radial_control_points = 8
    plot_results = True
    show_mesh = True

    # Random seed used for debugging between runs on the same mesh
    random.seed(37)

    # Create z and x points
    zpts = np.linspace(-draft, 0, radial_control_points)
    xpts = np.random.uniform(0, 5, size=(radial_control_points, 1))  # TODO: adjust bounds within constraints so that r(z) > 0
    xyz_pts = np.asarray([[xpts[i][0], 0, zpts[i]] for i in range(radial_control_points)])
    bez_x, bez_y, bez_pts = bezier_curve(xyz_pts, ntimes=100)

    buoy_shape, buoy_mesh, buoy_mass, buoy_stiffness = make_mesh(bez_pts)

    if show_mesh:
        buoy_mesh.show()

    buoy_hydrodynamic_data = evaluate_buoy_forces(buoy_mesh)
    power_data = complex_conjugate_control(buoy_hydrodynamic_data, buoy_mass, buoy_stiffness)
    annual_produced_power = objective_function(power_data)

    # Plot results
    if plot_results:
        plt.plot(omega_range, np.real(power_data), marker='o')
        plt.xlabel('$\omega$')
        plt.ylabel('Optimal Power')
        plt.tight_layout()
        plt.show()
        