import logging
import numpy as np
import capytaine as cpt
import matplotlib.pyplot as plt
from scipy.special import comb
import random
import xarray as xr
from scipy.optimize import minimize
import time

# logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t%(message)s")

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
    #approximate_resonance_radial_frequency = np.sqrt(stiffness / mass)

    #if verbose:
    #    print('\tMass = {}'.format(mass))
    #    print('\tStiffness = {}'.format(stiffness))
    #    print('\tApproximate resonance radial frequency = {} rad/s'.format(approximate_resonance_radial_frequency))

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

    # Set unrealistic negative radiation damping values to zero and add in extra system damping to counter potential flow assumptions
    radiation_damping[np.where(radiation_damping < 0.0)] = 0.0
    friction_damping = friction_damping_factor * np.max(radiation_damping)

    # Assemble unit wave amplitude excitation force in same format as added mass and radiation damping arrays
    device_data['excitation_force'] = device_data['Froude_Krylov_force'] + device_data['diffraction_force']
    excitation_force = device_data['excitation_force'].sel(wave_direction=wave_direction).data
    excitation_force = excitation_force.transpose()
    excitation_force = excitation_force[0, :]

    # Calculate the random wave offset and excitation for each frequency component
    wave_spectra = bretschneider_mitsuyasu_spectrum(omega_range, site_significant_wave_height,
                                                    site_significant_wave_period)
    d_omega = omega_range[1] - omega_range[0]
    np.random.seed(42)
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
        plt.plot(omega_range, radiation_damping, marker='o', label='Radiation Damping')
        plt.plot(omega_range, np.abs(excitation_force), marker='o', label='Excitation Force')
        # TODO: check np.abs function on excitation force magnitude

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


def objective_function(profile_points):

    # Create bezier points
    bez_x, bez_y, bez_pts = bezier_curve(profile_points, num_steps=20)

    # Make Mesh
    wec_shape, wec_mesh, wec_mass, wec_stiffness = make_mesh(bez_pts)

    if show_mesh:
        wec_mesh.show()

    # Evaluate mesh
    wec_hydrodynamic_data = evaluate_buoy_forces(wec_mesh)
    wec_power_data = complex_conjugate_control(wec_hydrodynamic_data, wec_mass, wec_stiffness)

    # Infinite penalty if any radii values are going too low or an unconstrained optimization method is going out of bounds
    if np.count_nonzero(bez_x < 0.01) > 0 or np.count_nonzero(bez_x > 2*draft) > 0 or np.count_nonzero(profile_points < 0.01) > 0 or np.count_nonzero(profile_points > 5) > 0:
        annual_power = 1e23
        # TODO: multiply infinite penalty by sum of constraint violations
    else:
        # Calculate Annual Power
        annual_power = -1.0 * np.sum(np.real(wec_power_data))

    # Plot power results    
    if plot_results:
        plt.plot(omega_range, np.real(wec_power_data), marker='o')
        plt.xlabel('$\omega$')
        plt.ylabel('Optimal Power')
        plt.tight_layout()
        plt.show()

    # Save results to global history variable
    point_history.append((profile_points, annual_power))

    if verbose:
        print('Current control points: {}'.format(profile_points),
              '\n\tAnnual Power: {}\n'.format(annual_power))

    return annual_power

def exhaustive_search(lower_bound, upper_bound, delta_x):

    d = len(delta_x)
    x_array_list = []
    n_candidates = 0

    for k in range(d):

        # Generate array of possible variable values along each dimension and add it to a list
        q_k = int((upper_bound[k] - lower_bound[k]) / delta_x[k]) + 1
        x_array = np.linspace(lower_bound[k], upper_bound[k], q_k)
        x_array_list.append(x_array)

        # Count the number of total design candidates
        if k == 0:
            n_candidates = q_k
        else:
            n_candidates = n_candidates * q_k

    # Initialize best current location and objective function value
    x_best = lower_bound
    f_best = objective_function(lower_bound)

    # Iteratively loop through all 4 dimensions
    for x1 in range(len(x_array_list[0])):
        for x2 in range(len(x_array_list[1])):
            x_new = np.array([x_array_list[0][x1], x_array_list[1][x2]])
            f_new = objective_function(x_new)

            if f_new < f_best:
                f_best = f_new
                x_best = np.copy(x_new)

    print('x* =\n', np.array_str(x_best, precision=3), '\nf* =', str(f_best))

    return x_best, f_best

def random_hill_climbing_algorithm(lower_bounds, upper_bounds, delta_x, random_seed):
    import random

    # Define transition function
    def random_climb(x, transitions, lower_bound, upper_bound):
        in_bounds = False
        while not in_bounds:
            rand_x_element = random.randint(0, len(x)-1)
            rand_climb_dir = random.choice(transitions)
            x_tmp = x[rand_x_element] + rand_climb_dir
            if (x_tmp >= lower_bound) and (x_tmp <= upper_bound):
                x[rand_x_element] = x_tmp
                in_bounds = True

        return np.array(x)

    # Find a random discrete starting point within the domain bounds if one is not given
    d = len(delta_x)
    if random_seed is not None:
        random.seed(random_seed)
    x_0 = np.zeros(shape=d)
    for k in range(d):
        n = int((upper_bounds[k] - lower_bounds[k]) / delta_x[k]) + 1
        x_0[k] = lower_bounds[k] + random.randint(0, n) * delta_x[k]

    # Define tunable parameters
    increment_options = np.array([0.50, -0.50, 0.25, -0.25])
    bounds = [0.01, 5.0]
    max_failed_moves = 5

    k = 0
    converged = False
    f_0 = objective_function(x_0)
    failed_moves = 0
    x_k = np.array(x_0)
    f_k = f_0
    while not converged:
        x_new = random_climb(x_k, increment_options, bounds[0], bounds[1])
        f_new = objective_function(profile_points=x_new)
        if f_new < f_k or np.isclose(f_new, f_k):
            x_k = np.array(x_new)
            f_k = f_new
            failed_moves = 0
        else:
            failed_moves += 1
            if failed_moves > max_failed_moves:
                converged = True
        k += 1

    print('x* =\n', np.array_str(x_k, precision=3), '\nf* =', str(f_k))


if __name__ == '__main__':

    ##############################
    # User inputs: WEC vars
    draft = 5
    ###radial_control_points = 2
    ###z_control_pts = np.linspace(-draft, 0, radial_control_points)
    mesh_resolution = 40
    mesh_bottom_cells = 6
    degree_of_freedom = 'Heave'
    friction_damping_factor = 0.10

    # User inputs: Wave vars
    wave_direction = 0.0
    omega_range = np.linspace(0.3, 2.0, 40)
    site_significant_wave_height = 3.5
    site_significant_wave_period = 11.0

    # User input: Visualization/Logging vars
    plot_results = False
    show_mesh = False
    verbose = True

    # User input: Optimization Vars
    ###opt_method = 'Nelder-Mead'
    number_of_runs = 1
    max_iterations = None  # Set to None if you want the default max iterations
    ##############################



    # Loop for if we want to run multiple times with different starting points
    for run in range(number_of_runs):
    
        # Global List used to record all profiles
        point_history = []

        # Random seed for reprodicible initial starting points
        ###np.random.seed(run * radial_control_points)
        
        if verbose:
            print('Starting Run {} of {}\n'.format(run + 1, number_of_runs))

        # Create starting radial control point
        ###start_x_points = np.random.uniform(0.01, 5, size=radial_control_points)

        # Create point bounds
        ###point_bounds = np.array([[0, 5]] * radial_control_points)

        # Run
        start_time = time.perf_counter()
        #result = minimize(objective_function,
        #                  start_x_points,
        #                  method=opt_method,
        #                  options={'disp': True, 'return_all': True, 'maxiter': max_iterations})
        #if run == 0:
        #    point_history = []
        #    radial_control_points = 2
        #    z_control_pts = np.linspace(-draft, 0, radial_control_points)
        #    opt_method = 'exhaustive_search'
        #    lower_bound = 0.25*np.ones(shape=2)
        #    upper_bound = 5.0*np.ones(shape=2)
        ##    delta_x = 0.25*np.ones(shape=2)
        #exhaustive_search(lower_bound=lower_bound, upper_bound=upper_bound, delta_x=delta_x)

        lower_bounds = 0.25*np.ones(shape=4)
        upper_bounds = 5.0*np.ones(shape=4)
        delta_x = 0.25*np.ones(shape=4)

        radial_control_points = 4
        z_control_pts = np.linspace(-draft, 0, radial_control_points)
        opt_method = 'random_hill_climbing_algorithm'
        random_hill_climbing_algorithm(lower_bounds=lower_bounds, upper_bounds=upper_bounds, delta_x=delta_x, random_seed=run)

        end_time = time.perf_counter()

        if verbose:
            print('\nTook {} minute(s) to run\n'.format((end_time - start_time) / 60))
        print(point_history)
        np.savez('./Run_{}_{}_{}_iter_{}_control'.format(run + 1, opt_method, max_iterations, radial_control_points), history=point_history)
                 #result=best_function_value,
