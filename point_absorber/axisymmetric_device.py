import logging
import numpy as np
import capytaine as cpt
import matplotlib.pyplot as plt
from scipy.special import comb
import random
import xarray as xr
from scipy.optimize import minimize
from geneticalgorithm import geneticalgorithm as ga
import time


# logging.basicConfig(level=logging.INFO,
#                     format="%(levelname)s:\t%(message)s")


# Generating the Bezier Curve:
# https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier_curve(points, num_steps=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        num_steps is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    # Separate the control points into individual coordinate arrays
    number_control_points = len(points)
    x_coord = points  # np.array([p for p in points])
    # y_coord = np.array([p[1] for p in points])
    z_coord = z_control_pts  # np.array([p[2] for p in points])

    # Create the polynomial array of the bezier curve that is of length num_steps
    t = np.linspace(0.0, 1.0, num_steps)
    polynomial_array = np.array(
        [bernstein_poly(i, number_control_points - 1, t) for i in range(0, number_control_points)])

    # Compute the new coordinates of the bezier curve and combine into one array
    x_vals = np.dot(x_coord, polynomial_array)
    # y_vals = np.dot(y_coord, polynomial_array)
    z_vals = np.dot(z_coord, polynomial_array)
    profile_pts = np.asarray([[x_vals[i], 0, z_vals[i]] for i in range(num_steps)])

    return x_vals, z_vals, profile_pts


def make_mesh(points_array):
    # Close the bottom of the points
    bottom = np.array(points_array[0])
    bottom_spacing = np.linspace(0, bottom[0], mesh_bottom_cells)
    bottom_pts = np.asarray([[bottom_spacing[i], 0, bottom[2]] for i in range(mesh_bottom_cells - 1)])
    mesh_shape = np.concatenate((bottom_pts, points_array), axis=0)

    # Make mesh and add the vertical degree of freedom
    buoy = cpt.FloatingBody(cpt.AxialSymmetricMesh.from_profile(profile=mesh_shape, nphi=mesh_resolution))
    buoy.add_translation_dof(name=degree_of_freedom)

    # Calculate the mass and stiffness parameters
    volume = buoy.mesh.volume
    mass = volume * 1000
    radius = points_array[-1][0]
    stiffness = np.pi * 1000 * 9.81 * radius ** 2

    return mesh_shape, buoy, mass, stiffness


def evaluate_buoy_forces(buoy_object):
    # Set up radiation and diffraction problems
    problems = [cpt.RadiationProblem(body=buoy_object, radiating_dof=degree_of_freedom, omega=omega)
                for omega in omega_range]
    problems += [cpt.DiffractionProblem(omega=omega, body=buoy_object, wave_direction=wave_direction)
                 for omega in omega_range]

    # Solve each matrix problem
    solver = cpt.BEMSolver(engine=cpt.HierarchicalToeplitzMatrixEngine())
    results = [solver.solve(pb) for pb in sorted(problems)]

    return cpt.assemble_dataset(results)


def complex_conjugate_control(device_data, device_mass, device_stiffness):
    # Assemble needed values for control algorithm
    added_mass = device_data['added_mass'].sel(radiating_dof=degree_of_freedom, influenced_dof=degree_of_freedom).data
    radiation_damping = device_data['radiation_damping'].sel(radiating_dof=degree_of_freedom,
                                                             influenced_dof=degree_of_freedom).data
    friction_damping = 0.10 * np.max(radiation_damping)
    # TODO: adjust frictional damping if necessary

    # Assemble unit wave amplitude excitation force in same format as added mass and radiation damping arrays
    device_data['excitation_force'] = device_data['Froude_Krylov_force'] + device_data['diffraction_force']
    excitation_force = device_data['excitation_force'].sel(wave_direction=wave_direction).data
    excitation_force = excitation_force.transpose()
    excitation_force = excitation_force[0, :]

    # Calculate the random wave offset and excitation for each frequency component
    wave_spectra = bretschneider_mitsuyasu_spectrum(omega_range, site_significant_wave_height,
                                                    site_significant_wave_period)
    d_omega = omega_range[1] - omega_range[0]
    random_phase_offset = 2 * np.pi * np.random.rand(len(omega_range))
    frequency_domain_wave_spectrum_amplitudes = 2 * np.sqrt(wave_spectra * d_omega) * np.exp(1.0j * random_phase_offset)
    excitation_force = excitation_force * frequency_domain_wave_spectrum_amplitudes

    # Complex conjugate control algorithm 
    intrinsic_impedance = (radiation_damping + friction_damping) + 1.0j * omega_range * \
                          ((device_mass + added_mass - device_stiffness) / (omega_range ** 2))
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


def objective_function2(profile_points):
    global time_elapsed

    # Create bezier points
    bez_x, bez_y, bez_pts = bezier_curve(profile_points, num_steps=100)

    # Make Mesh
    wec_shape, wec_mesh, wec_mass, wec_stiffness = make_mesh(bez_pts)

    if show_mesh:
        wec_mesh.show()

    # Evaluate mesh
    wec_hydrodynamic_data = evaluate_buoy_forces(wec_mesh)
    wec_power_data = complex_conjugate_control(wec_hydrodynamic_data, wec_mass, wec_stiffness)

    # Calculate Annual Power
    annual_power = -1.0 * np.sum(np.real(wec_power_data))
    # print(annual_power)

    # Plot results
    if plot_results:
        plt.plot(omega_range, np.real(wec_power_data), marker='o')
        plt.xlabel('$\omega$')
        plt.ylabel('Optimal Power')
        plt.tight_layout()
        plt.show()

    # Save results to global history variable
    point_history.append([profile_points, annual_power])

    if verbose:
        if (time.perf_counter()-start_time)/60 > time_elapsed:
            print('\t{} minutes elapsed'.format(round(((time.perf_counter()-start_time)/60), 2)),
                  '\n\tCurrent control points: {}'.format(profile_points),
                  '\n\tAnnual Power: {}\n'.format(annual_power)
                  )
            time_elapsed += print_freq

    return annual_power


if __name__ == '__main__':

    ##############################
    # User inputs: WEC vars
    draft = 5
    radial_control_points = 8
    z_control_pts = np.linspace(-draft, 0, radial_control_points)
    mesh_resolution = 40
    mesh_bottom_cells = 6
    degree_of_freedom = 'Heave'

    # User inputs: Wave vars
    wave_direction = 0.0
    omega_range = np.linspace(0.3, 2.0, 40)
    site_significant_wave_height = 3.5
    site_significant_wave_period = 11.0

    # User input: Visualization/Logging vars
    plot_results = False
    show_mesh = False
    verbose = True
    print_freq = 5  # minutes

    # User input: Optimization Vars
    opt_method = 'Nelder-Mead'
    number_of_runs = 1
    max_iterations = 1  # Set to None if you want the default max iterations
    ##############################

    # Global List used to record all profiles
    point_history = []

    # Random seed used for debugging between runs on the same mesh
    # random.seed(37)

    # Loop for if we want to run multiple times with different starting points
    for run in range(number_of_runs):
        time_elapsed = print_freq

        if verbose:
            print('Starting Run {} of {}\n'.format(run+1, number_of_runs))

        # Create starting radial control point
        start_x_points = np.random.uniform(0.01, 5, size=radial_control_points)

        # Creat point bounds
        point_bounds = np.array([[0, 5]] * radial_control_points)

        # Run
        start_time = time.perf_counter()
        result = minimize(objective_function2,
                          start_x_points,
                          method=opt_method,
                          options={'disp': True, 'return_all': True, 'maxiter': max_iterations}
                          )
        end_time = time.perf_counter()

        if verbose:
            print('\nTook {} minute(s) to run\n'.format((end_time-start_time)/60), result)

        np.savez('./Run_{}_Results_{}_{}_controlpts'.format(run+1, opt_method, radial_control_points),
                   result=result,
                   history=point_history
                   )

    # Create the GA model and run it
    # model_int = ga(function=objective_function2, dimension=8, variable_type='real', variable_boundaries=point_bounds)
    # model_int.run()
