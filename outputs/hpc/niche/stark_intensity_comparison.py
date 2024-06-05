from dataclasses import replace
import matplotlib.pyplot as plt

from outputs.hpc.run_everything import compute_spectra

STATIC_ARGS = {
    "temperature_kelvin": 300,
    "mode_frequencies": [1700, 100],
    "mode_couplings": [0.7, 8.0],
    "transfer_integral": 100,
    "broadening": 200,
    "mode_basis_sets": [20, 200],
    "disorder_meV": 0,
    "basis_size": 20,
    "positive_field_sum_percent": 0.5,
    "positive_field_strength": 0.1,
    "field_delta_dipole": 38,
    "field_delta_polarizability": 0,
}

ENERGY_GAP_LABEL = "energy_gap"
ENERGY_GAPS = [0, 200, 400, 800]

MIN_X_WAVENUMBERS = 1000
MAX_X_WAVENUMBERS = None

MODEL = "mlj"

max_reference_y = None

for energy_gap in ENERGY_GAPS:
    args = STATIC_ARGS.copy()
    args[ENERGY_GAP_LABEL] = energy_gap

    _, stark_spectra = compute_spectra(args, force_loading=True, model=MODEL)

    stark_spectra = stark_spectra.cut_bounds(
        start_energy=MIN_X_WAVENUMBERS, end_energy=MAX_X_WAVENUMBERS
    )

    # if another spectrum has already been normalized,
    # this ensures that relative intensities are preserved
    max_y = max(stark_spectra.intensities)
    percentage_of_reference = None

    if max_reference_y is None:
        max_reference_y = max_y

    percentage_of_reference = max_y / max_reference_y  # is scaling necessary here?

    plt.plot(
        stark_spectra.energies,
        stark_spectra.intensities,
        label=f"Energy gap: {energy_gap}",
    )

plt.title(f"{MODEL} comparison")

plt.xlabel(r"$cm^{-1}$")
plt.ylabel(r"$\Delta E$")

plt.legend()
plt.show()
