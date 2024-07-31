from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from math import factorial
from typing import Literal

parent = Path(__file__).parent

from quantumspectra_2024.models.three_state.ThreeStateSimpleModel import (
    ThreeStateSimpleModel,
)

from quantumspectra_2024.models.two_state.TwoStateComputation import broaden_peaks

from quantumspectra_2024.common.hamiltonian.HamiltonianComputation import (
    calculate_state_local_diagonals,
)

model = ThreeStateSimpleModel(
    temperature_kelvin=300,
    ct_energy_gap=10745,
    le_energy_gap=10745,
    gs_ct_coupling=40,
    ct_le_coupling=0,
    le_mode_couplings=[np.sqrt(0.9), np.sqrt(0.5)],
    ct_mode_couplings=[0, 0],
    mode_basis_sets=[10, 5],
    mode_frequencies=[1400, 100],
)

# Whether to create a distribution from just CT or LE state, or both
USE_PEAKS_FROM_STATE: Literal["CT", "LE", "Both"] = "LE"

# When comparing peaks to the S ratio, whether to use the first or second basis set
# the first set should be the large one (bigger number)
COMPARE_PEAKS_FROM_SET: Literal["Large", "Small", "Both"] = "Both"

# display which peaks where selected from all generated
SHOW_SELECTED_PEAKS = True

# display ratios with n as the x axis
SHOW_SELECTED_RATIOS = True

# how many selected peaks to compare
MAX_PEAKS_TO_COMPARE = 10

# manually set a window size of wavenumbers to compare states
# starts from the first peak in each state and adds the window size
# if set to None, will separate by #mode_basis_sets[1] peaks
STATE_PEAK_SIZE_WAVENUBMERS = 1500


# ----- START -------

hamiltonian = model.get_hamiltonian()

vals, vects = hamiltonian.get_diagonalization()

ground_state_energies = calculate_state_local_diagonals(
    state_energy=0,
    mode_frequencies=model.mode_frequencies,
    mode_couplings=[0] * len(model.mode_basis_sets),
    mode_basis_sets=model.mode_basis_sets,
)

temperature_wavenumbers = model.temperature_kelvin * 0.695028

block_size = np.prod(model.mode_basis_sets)


def compute_state_energies_intensities(state: Literal["CT", "LE"]):
    idx_offset = 0 if state == "CT" else block_size

    energy_groups = []
    intensity_groups = []

    for idx, energy in enumerate(ground_state_energies[:1]):
        offset_idx = idx + idx_offset

        new_energies = vals[idx_offset : idx_offset + block_size] - energy
        new_intensities = (vects[offset_idx, idx_offset : idx_offset + block_size]) ** 2

        energy_groups.append(new_energies)
        intensity_groups.append(new_intensities)

    energy_groups = np.array(energy_groups)
    intensity_groups = np.array(intensity_groups)

    return energy_groups, intensity_groups


# Thermal population
def compute_thermal_population():
    ground_state_diffs = ground_state_energies - ground_state_energies[0]
    ground_state_exps = np.exp(-ground_state_diffs / temperature_wavenumbers)
    return ground_state_exps / np.sum(ground_state_exps)


thermal_population = compute_thermal_population()


peak_energies = np.array([])
peak_intensities = np.array([])


# compute a group of values and append to the peak arrays
def append_groups(state: Literal["CT", "LE"]):
    e_groups, i_groups = compute_state_energies_intensities(state)

    new_peak_energies = np.append(peak_energies, e_groups.flatten())
    new_peak_intensities = np.append(peak_intensities, i_groups.flatten())

    return new_peak_energies, new_peak_intensities


if USE_PEAKS_FROM_STATE == "CT" or USE_PEAKS_FROM_STATE == "Both":
    peak_energies, peak_intensities = append_groups("CT")

if USE_PEAKS_FROM_STATE == "LE" or USE_PEAKS_FROM_STATE == "Both":
    peak_energies, peak_intensities = append_groups("LE")


# ---- PEAKS GENERATED ----


# sorts such that x values are in ascending order
def sort_x_y(x, y):
    x, y = zip(*sorted(zip(x, y)))
    return np.array(x), np.array(y)


# peak sorting
peak_energies, peak_intensities = sort_x_y(peak_energies, peak_intensities)


# PEAK SELECTION
def select_peaks(set: Literal["Large", "Small"]):
    comparison_slice_fn = None

    if set == "Large":

        def slice_large_peaks(arr):
            selected_peaks = arr[:: model.mode_basis_sets[1]]
            mask = np.ones(selected_peaks.shape, dtype=bool)
            mask[2::2] = False  # mask out every other element from idx 2
            return selected_peaks[mask]

        comparison_slice_fn = lambda arr: slice_large_peaks(arr)

    elif set == "Small":
        comparison_slice_fn = lambda arr: arr[: model.mode_basis_sets[1]]

    comparison_energies = comparison_slice_fn(peak_energies)
    comparison_intensities = comparison_slice_fn(peak_intensities)

    if SHOW_SELECTED_PEAKS:
        plt.scatter(peak_energies, peak_intensities, label="All")
        plt.scatter(comparison_energies, comparison_intensities, label="Selected")
        plt.title(f"Selected peaks w/ SET = {set}")
        plt.legend()
        plt.show()

    return comparison_energies, comparison_intensities


# RATIO COMPARING
# right now, assumed selected arrays come sorted
def get_ratios(set: Literal["Large", "Small"], comparison_intensities):
    ratios = []
    expected_ratios = []
    counts = []
    max_y = comparison_intensities[0]

    # find S value
    coupling_value_idx = 0 if set == "Large" else 1
    coupling_value = model.ct_mode_couplings[coupling_value_idx]
    S = coupling_value**2

    for count, comparison_y in enumerate(comparison_intensities[1:]):

        def get_expected_ratio_value():
            n = count + 1
            return (S**n) / factorial(n)

        ratio = comparison_y / max_y
        expected_ratio = get_expected_ratio_value()

        print(f"expected: {expected_ratio:.2f} actual: {ratio:.2f}")

        ratios.append(ratio)
        expected_ratios.append(expected_ratio)
        counts.append(count)

        if count > MAX_PEAKS_TO_COMPARE:
            break

    if SHOW_SELECTED_RATIOS:
        plt.scatter(counts, ratios, label="Actual")
        plt.scatter(counts, expected_ratios, label="Expected", alpha=0.7)
        plt.title(f"Peak intensity ratios w/ S={S:.2f} and SET={set}")
        plt.legend()
        plt.show()

    return counts, ratios, expected_ratios, S


sets_to_compare: list[Literal["Large", "Small"]] = []

if COMPARE_PEAKS_FROM_SET == "Large" or COMPARE_PEAKS_FROM_SET == "Both":
    sets_to_compare.append("Large")

if COMPARE_PEAKS_FROM_SET == "Small" or COMPARE_PEAKS_FROM_SET == "Both":
    sets_to_compare.append("Small")

for set in sets_to_compare:
    comparison_energies, comparison_intensities = select_peaks(set)
    get_ratios(set, comparison_intensities)
