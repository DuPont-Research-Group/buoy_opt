def bretschneider_mitsuyasu_spectrum(wave_radial_frequency_array, significant_wave_height, significant_wave_period):
    f = wave_radial_frequency_array / (2 * np.pi)

    h = significant_wave_height
    t = significant_wave_period

    spectra = 0.257 * (h ** 2) * (t ** -4) * (f ** -5) * np.exp(-1.03 * (t * f) ** -4)

    return spectra


import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)
omega_range = np.linspace(0.3, 2.0, 300)
spectra = bretschneider_mitsuyasu_spectrum(omega_range, significant_wave_height=3.5, significant_wave_period=11.0)
d_omega = omega_range[1] - omega_range[0]

wave_spectrum_amplitudes = 2 * np.sqrt(spectra * d_omega)

eta = 0
t = np.linspace(0, 100, 1000)
for k in range(len(omega_range)):
    eta += wave_spectrum_amplitudes[k] * np.cos(omega_range[k]*t + 2*np.pi*random.random())

plt.plot(omega_range, spectra)
plt.xlabel('$\omega$')
plt.ylabel(('$S(\omega)$'))
plt.show()

plt.plot(omega_range, wave_spectrum_amplitudes)
plt.xlabel('$\omega$')
plt.ylabel(('$a(\omega)$'))
plt.show()

plt.plot(t, eta)
plt.xlabel('$t [s]$')
plt.ylabel('$\eta(t)$')
plt.show()
