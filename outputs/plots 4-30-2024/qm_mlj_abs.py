import matplotlib.pyplot as plt
from quantumspectra_2024.models import TwoStateModel, MLJModel
from plot_utils import (
    save_file,
    open_file,
    match_spectrum_greatest_peak,
    scale_relative_to_one,
    wavenumbers_to_ev,
    ABS_YLABEL,
    ENERGY_XLABEL,
    TITLEDICT,
    LABELDICT,
    LEGENDDICT,
    LINEWIDTH,
)

# SLIDE 1 - QM PLOT ON LEFT, MLJ ON RIGHT - COMPARES G2 VALUES

REGENERATE = False
PICKLE_PREFIX = "pickles/qm_mlj_abs/"

COMMON_ARGS = {
    "temperature_kelvin": 300,
    "energy_gap": 8_000,
    "mode_frequencies": [1200, 100],
    "mode_couplings": [0.7, 2.0],
}

TWO_STATE_ARGS = {
    "broadening": 200,
    "transfer_integral": 100,
    "mode_basis_sets": [20, 200],
}

MLJ_ARGS = {
    "disorder_meV": 0,
    "basis_size": 20,
}

COUPLING_G2_VALS = [2.0, 4.0, 6.0, 8.0]

ENERGY_LOW_BOUND = 6_000  # cm-1, or None
ENERGY_HIGH_BOUND = 20_000  # cm-1, or None


def get_spectra():
    if REGENERATE:
        mlj_spectra = compute_spectra(MLJModel, MLJ_ARGS)
        two_state_spectra = compute_spectra(TwoStateModel, TWO_STATE_ARGS)
        # save spectra
        save_file(two_state_spectra, f"{PICKLE_PREFIX}two_state_spectra.pkl")
        save_file(mlj_spectra, f"{PICKLE_PREFIX}mlj_spectra.pkl")
    else:
        two_state_spectra = open_file(f"{PICKLE_PREFIX}two_state_spectra.pkl")
        mlj_spectra = open_file(f"{PICKLE_PREFIX}mlj_spectra.pkl")

    return two_state_spectra, mlj_spectra


def compute_spectra(model, args):
    model_args = {**COMMON_ARGS, **args}

    spectra = []
    for g2 in COUPLING_G2_VALS:
        model_args["mode_couplings"][1] = g2
        model_instance = model(**model_args)
        spectrum = model_instance.get_absorption()
        spectra.append(spectrum)
    return spectra


def configure_axes(ax, title, ylabel):
    ax.set_title(title, TITLEDICT)
    ax.set_ylabel(ylabel, LABELDICT)

    ax.set_xlabel(ENERGY_XLABEL, LABELDICT)

    ax.set_xlim(
        wavenumbers_to_ev(ENERGY_LOW_BOUND), wavenumbers_to_ev(ENERGY_HIGH_BOUND)
    )

    ax.legend(prop=LEGENDDICT)


def main():
    two_state_spectra, mlj_spectra = get_spectra()

    # matching scaling
    for idx, (two_state_spectrum, mlj_spectrum) in enumerate(
        zip(two_state_spectra, mlj_spectra)
    ):
        mlj_spectra[idx] = match_spectrum_greatest_peak(
            two_state_spectrum, mlj_spectrum
        )

    # scaling relative to one
    scaled_spectra = scale_relative_to_one(two_state_spectra + mlj_spectra)

    two_state_spectra = scaled_spectra[: len(two_state_spectra)]
    mlj_spectra = scaled_spectra[len(two_state_spectra) :]

    _, ax = plt.subplots(1, 2)

    for spectrum, g2 in zip(two_state_spectra, COUPLING_G2_VALS):
        ax[0].plot(
            wavenumbers_to_ev(spectrum.energies),
            spectrum.intensities,
            label=rf"$g_2 = {g2:.0f}$",
            linewidth=LINEWIDTH,
        )

    for spectrum, g2 in zip(mlj_spectra, COUPLING_G2_VALS):
        ax[1].plot(
            wavenumbers_to_ev(spectrum.energies),
            spectrum.intensities,
            label=rf"$g_2 = {g2:.0f}$",
            linewidth=LINEWIDTH,
        )

    configure_axes(ax[0], "QM Absorption", ABS_YLABEL)
    configure_axes(ax[1], "MLJ Absorption", "")

    plt.show()


if __name__ == "__main__":
    main()
