from pathlib import Path
import numpy as np
from math import factorial
import matplotlib.pyplot as plt

parent = Path(__file__).parent

from quantumspectra_2024.models.three_state.ThreeStateSimpleModel import (
    ThreeStateSimpleModel,
)

from quantumspectra_2024.models.two_state.TwoStateComputation import (
    broaden_peaks,
)

from quantumspectra_2024.common.hamiltonian.HamiltonianComputation import (
    calculate_state_local_diagonals,
)

model = ThreeStateSimpleModel(
    temperature_kelvin=300,
    ct_energy_gap=10745,
    le_energy_gap=12358,
    gs_ct_coupling=40,
    ct_le_coupling=0,
    ct_mode_couplings=[np.sqrt(0.9), np.sqrt(0.5)],
    le_mode_couplings=[0, 0],
    mode_basis_sets=[10, 5],
    mode_frequencies=[1400, 100],
)

hamiltonian = model.get_hamiltonian()

vals, vects = hamiltonian.get_diagonalization()

# np.savetxt(parent / "new_eigenvals.dat", vals)
# np.savetxt(parent / "new_eigenvects.dat", vects)

ground_state_energies = calculate_state_local_diagonals(
    0, model.mode_frequencies, [0, 0], model.mode_basis_sets
)

temperature_wavenumbers = model.temperature_kelvin * 0.695028

block_size = np.prod(model.mode_basis_sets)


def compute_energies_intensities(idx_offset):
    energy_groups = []
    intensity_groups = []

    for idx, energy in enumerate(ground_state_energies[:1]):

        offset_idx = idx + idx_offset

        new_energies = vals[idx_offset : idx_offset + block_size] - energy
        new_intensities = (
            vects[offset_idx, idx_offset : idx_offset + block_size] * 100
        ) ** 2

        energy_groups.append(new_energies)
        intensity_groups.append(new_intensities)

    energy_groups = np.array(energy_groups)
    intensity_groups = np.array(intensity_groups)

    return energy_groups, intensity_groups


ct_energy_groups, ct_intensity_groups = compute_energies_intensities(0)
le_energy_groups, le_intensity_groups = compute_energies_intensities(block_size)

# ------------------------------------------------------------
# to compute a thermal population, find the difference of each ground state energy from the first
ground_state_differences = ground_state_energies - ground_state_energies[0]
ground_state_exponentials = np.exp(-ground_state_differences / temperature_wavenumbers)
thermal_population = ground_state_exponentials / np.sum(ground_state_exponentials)
# ------------------------------------------------------------

peak_energies = ct_energy_groups.flatten()
peak_intensities = np.concatenate(
    (
        (ct_intensity_groups * 1).flatten(),
        # (le_intensity_groups * 1).flatten(),
    )
)

sample_points = np.linspace(model.start_energy, model.end_energy, model.num_points)

peak_intensities = peak_intensities[:5]

# comparing peak intensities in sorted order
sorted_peak_intensity_indices = np.argsort(peak_intensities)[::-1]

max_intensity = peak_intensities[sorted_peak_intensity_indices[0]]
ratios = []
counts = []

for count, comparison_idx in enumerate(sorted_peak_intensity_indices[1:]):

    def get_expected_ratio_value():
        coupling_value = model.ct_mode_couplings[1]
        S = coupling_value**2
        n = count + 1
        return (S**n) / factorial(n)

    comparison_intensity = peak_intensities[comparison_idx]
    ratio = comparison_intensity / max_intensity

    print(f"expected: {get_expected_ratio_value():.2f} actual: {ratio:.2f}")

    ratios.append(ratio)
    counts.append(count)

    if count > 10:
        break

plt.scatter(counts, ratios)
plt.show()


plt.scatter(peak_energies, peak_intensities)
plt.show()

import sys

sys.exit()

broadened_peaks = broaden_peaks(sample_points, peak_energies, peak_intensities, 300)

fortran_data = np.loadtxt(parent / "TwoState_absorption_1.dat")

peak_broadened = np.max(broadened_peaks)
peak_fortran = np.max(fortran_data[:, 1])

scaling_factor = peak_broadened / peak_fortran

plt.plot(sample_points, broadened_peaks, label="python")


plt.plot(fortran_data[:, 0], fortran_data[:, 1] * scaling_factor, label="fortran")

plt.legend()
plt.show()
