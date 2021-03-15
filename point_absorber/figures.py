def bretschneider_mitsuyasu_spectrum(wave_radial_frequency_array, significant_wave_height, significant_wave_period):
    f = wave_radial_frequency_array / (2 * np.pi)

    h = significant_wave_height
    t = significant_wave_period

    spectra = 0.257 * (h ** 2) * (t ** -4) * (f ** -5) * np.exp(-1.03 * (t * f) ** -4)

    return spectra


import numpy as np
import matplotlib.pyplot as plt
import random

#random.seed(42)
#omega_range = np.linspace(0.3, 2.0, 300)
omega_range = np.linspace(0.3, 2.0, 40)

spectra = bretschneider_mitsuyasu_spectrum(omega_range, significant_wave_height=3.5, significant_wave_period=11.0)
d_omega = omega_range[1] - omega_range[0]

wave_spectrum_amplitudes = 2 * np.sqrt(spectra * d_omega)

eta = 0
t = np.linspace(0, 100, 1000)
for k in range(len(omega_range)):
    eta += wave_spectrum_amplitudes[k] * np.cos(omega_range[k]*t + 2*np.pi*random.random())

#plt.plot(omega_range, spectra)
#plt.xlabel('$\omega \ (rad/s)$')
#plt.ylabel(('$S(\omega) $'))
#plt.show()

plt.bar(omega_range, wave_spectrum_amplitudes, width=omega_range[1] - omega_range[0])
plt.xlabel('$\omega $ (rad/s)')
plt.ylabel(('$a(\omega) $ (m)'))
plt.xlim((0.0, 2.1))
plt.show()

#plt.plot(t, eta)
#plt.xlabel('$t [s]$')
#plt.ylabel('$\eta(t)$')
#plt.show()


import logging
import numpy as np
import capytaine as cpt
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
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
    excitation_force_unweighted = excitation_force

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

    return optimal_power, added_mass, radiation_damping, excitation_force_unweighted, excitation_force, intrinsic_impedance


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
    wec_power_data, mass, damping, ex_w, ex, impedance = complex_conjugate_control(wec_hydrodynamic_data, wec_mass, wec_stiffness)

    # Infinite penalty if any radii values are going too low or an unconstrained optimization method is going out of bounds
    if np.count_nonzero(bez_x < 0.01) > 0 or np.count_nonzero(bez_x > 2*draft) > 0 or np.count_nonzero(profile_points < 0.01) > 0 or np.count_nonzero(profile_points > 5) > 0:
        annual_power = 1e23
        # TODO: multiply infinite penalty by sum of constraint violations
    else:
        # Calculate Annual Power
        annual_power = -1.0 * np.sum(np.real(wec_power_data))

    if verbose:
        print('Current control points: {}'.format(profile_points),
              '\n\tAnnual Power: {}\n'.format(annual_power))

    return annual_power, mass, damping, ex_w, ex, impedance



if __name__ == '__main__':

    ##############################
    # User inputs: WEC vars
    draft = 5
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
    plot_results = True
    show_mesh = False
    verbose = True

    radial_control_points = 4
    z_control_pts = np.linspace(-draft, 0, radial_control_points)

    designs = [np.array([5.0, 5.0, 5.0, 5.0]), np.array([4.5, 2.5, 0.75, 2.75])]
    design_count = len(designs)

    k = 0
    for design_variables in designs:
        power, added_mass, radiation_damping, excitation_force_unweighted, excitation_force_weighted, intrinsic_impedance = objective_function(design_variables)

        if plot_results:
            fig_index = 221 + k
            plt.subplot(fig_index)
            plt.plot(omega_range, added_mass, marker='o', label='Added Mass')
            plt.plot(omega_range, radiation_damping, marker='o', label='Radiation Damping')
            plt.plot(omega_range, np.abs(excitation_force_unweighted), marker='o', label='Wave Unmodified Excitation Force')        
            plt.plot(omega_range, np.abs(excitation_force_weighted), marker='o', label='Wave Modified Excitation Force')
            plt.xlabel('$\omega$ (rad/s)')
            if k == 0:
                plt.ylim((0, 1000000))
                plt.legend()
            else:
                plt.ylim((0, 250000))
            #plt.tight_layout()
            plt.grid()

            fig_index = 223 + k
            plt.subplot(fig_index)
            plt.plot(omega_range, np.abs(intrinsic_impedance), marker='o')
            plt.xlabel('$\omega$ (rad/s)')
            plt.ylabel('System Intrinsic Impedance Magnitude')
            plt.ylim((0, 350000))
            #plt.tight_layout()

            #plt.subplot()
            #plt.plot(omega_range, np.angle(intrinsic_impedance, deg=True), marker='o', label='Power Take Off Impedance')
            #plt.plot(omega_range, np.abs(power_take_off_force), marker='o', label='Power Take Off Force')
            #plt.xlabel('$\omega$ (rad/s)')
            #plt.ylabel('System Intrinsic Impedance Complex Angle')
            #plt.tight_layout()
            #plt.show()
        
        k += 1
    plt.show()