import os
import logging
import numpy as np
import capytaine as cpt

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:\t%(message)s")

# Inputs
draft = 5
show_mesh = False
device_mass = 1e5
device_stiffness = 1e4

os.system('cls')

# Set up shape profile for the axisymmetric body
def shape(z):
    return 0.1*(-(z+1)**2 + 16)


buoy = cpt.FloatingBody(
    cpt.AxialSymmetricMesh.from_profile(shape, z_range=np.linspace(-draft, 0, 30), nphi=40)
)
buoy.add_translation_dof(name="Heave")

if show_mesh:
    buoy.show()

# Set up radiation and diffraction problems
omega_range = np.linspace(0.3, 5.0, 60)
problems = [cpt.RadiationProblem(body=buoy, radiating_dof='Heave', omega=omega)
            for omega in omega_range]
problems += [cpt.DiffractionProblem(omega=omega, body=buoy, wave_direction=0.0)
            for omega in omega_range]


# Solve the problems using the axial symmetry
solver = cpt.BEMSolver(engine=cpt.HierarchicalToeplitzMatrixEngine())
results = [solver.solve(pb) for pb in sorted(problems)]
*radiation_results, diffraction_result = results
dataset = cpt.assemble_dataset(results)

# Plot results
import matplotlib.pyplot as plt

added_mass = dataset['added_mass'].sel(radiating_dof='Heave', influenced_dof='Heave')
radiation_damping = dataset['radiation_damping'].sel(radiating_dof='Heave', influenced_dof='Heave')

dataset['excitation_force'] = dataset['Froude_Krylov_force'] + dataset['diffraction_force']
excitation_force = dataset['excitation_force'].sel(wave_direction=0.0)

intrinsic_impedance = radiation_damping + 1.0j * omega_range * (device_mass + added_mass - device_stiffness / (omega_range ** 2))
power_take_off_impedance = np.conjugate(intrinsic_impedance)

optimal_velocity = excitation_force / (2 * np.real(intrinsic_impedance))

power_take_off_force = power_take_off_impedance * optimal_velocity
optimal_power = np.real(power_take_off_impedance * optimal_velocity)


plt.figure()
plt.plot(omega_range, optimal_power, marker='o')
plt.xlabel('omega')
plt.ylabel('Optimal Power')
plt.tight_layout()
plt.show()
