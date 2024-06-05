from dataclasses import replace
import matplotlib.pyplot as plt

from outputs.hpc.run_everything import compute_spectra
from outputs.hpc.experiment.sample_experiment import (
    x as x_expr,
    y_smooth as y_expr_smooth,
)

STATIC_ARGS = {
    "temperature_kelvin": 300,
    "mode_frequencies": [1700, 100],
    "mode_couplings": [1.5, 6.67],
    "transfer_integral": 100,
    "broadening": 200,
    "mode_basis_sets": [20, 200],
    "disorder_meV": 0,
    "basis_size": 20,
    "positive_field_sum_percent": 0.5,
    "positive_field_strength": 0.0,
    "field_delta_dipole": 0,
    "field_delta_polarizability": 0,
}

ENERGY_GAP_LABEL = "energy_gap"
ENERGY_GAPS = [0, 200, 400, 800]

MIN_X_WAVENUMBERS = 1_500
MAX_X_WAVENUMBERS = 10_000

SHOW_EXPERIMENT = True
SHOW_TWO_STATE = True


if SHOW_EXPERIMENT:
    plt.plot(x_expr, y_expr_smooth, label="Experiment")

max_expr_y = max(y_expr_smooth)
max_reference_y = None

for energy_gap in ENERGY_GAPS:
    args = STATIC_ARGS.copy()
    args[ENERGY_GAP_LABEL] = energy_gap

    two_state_spectrum, _, _, _ = compute_spectra(args)

    two_state_spectrum = two_state_spectrum.cut_bounds(
        start_energy=MIN_X_WAVENUMBERS, end_energy=MAX_X_WAVENUMBERS
    )

    # if another spectrum has already been normalized,
    # this ensures that relative intensities are preserved
    max_y = max(two_state_spectrum.intensities)
    percentage_of_reference = None

    if max_reference_y is None:
        max_reference_y = max_y

    percentage_of_reference = max_y / max_reference_y

    # perform scaling
    scaling_factor = (max_expr_y / max_y) * percentage_of_reference

    two_state_spectrum = replace(
        two_state_spectrum, intensities=two_state_spectrum.intensities * scaling_factor
    )

    if SHOW_TWO_STATE:
        plt.plot(
            two_state_spectrum.energies,
            two_state_spectrum.intensities,
            label=f"Energy gap: {energy_gap}",
        )

plt.xlabel(r"$cm^{-1}$")
plt.ylabel(r"$\Delta E$")

if SHOW_TWO_STATE:
    plt.title("QM Two-State")

plt.legend()
plt.show()
