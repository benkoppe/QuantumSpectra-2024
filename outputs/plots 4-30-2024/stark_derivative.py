import numpy as np
import inspect
import functools
import matplotlib.pyplot as plt

from quantumspectra_2024.models import TwoStateModel, MLJModel, StarkModel
from plot_utils import save_file, open_file, get_stats, fix_twinx_ticks

REGENERATE = False
PICKLE_PREFIX = "pickles/stark_derivatives"

COMMON_ARGS = {
    "temperature_kelvin": 300,
    "energy_gap": 8_000,
    "mode_frequencies": [1200, 100],
    "mode_couplings": [0.7, 2.0],
}

TWO_STATE_ARGS = {
    "transfer_integral": 100,
    "broadening": 200,
    "mode_basis_sets": [20, 200],
}

MLJ_ARGS = {
    "disorder_meV": 0,
    "basis_size": 20,
}

STARK_ARGS = {
    "positive_field_sum_percent": 0.5,
    "positive_field_strength": 0.01,
}

DELTA_DIPOLE = 38
DELTA_POLARIZABILITY = 1_000


def compute_spectra():
    two_state_valids = list(inspect.signature(TwoStateModel).parameters)
    mlj_valids = list(inspect.signature(MLJModel).parameters)
    stark_valids = list(inspect.signature(StarkModel).parameters)

    all_args = {**COMMON_ARGS, **TWO_STATE_ARGS, **MLJ_ARGS, **STARK_ARGS}

    two_state_args = {k: v for k, v in all_args.items() if k in two_state_valids}
    mlj_args = {k: v for k, v in all_args.items() if k in mlj_valids}
    stark_args = {k: v for k, v in all_args.items() if k in stark_valids}

    two_state_model = TwoStateModel(**two_state_args)
    mlj_model = MLJModel(**mlj_args)

    TwoStateStarkModel = functools.partial(
        StarkModel, neutral_submodel=two_state_model, **stark_args
    )
    two_state_stark_model_dipole = TwoStateStarkModel(
        field_delta_dipole=DELTA_DIPOLE, field_delta_polarizability=0.0
    )
    two_state_stark_model_polarizability = TwoStateStarkModel(
        field_delta_dipole=0.0, field_delta_polarizability=DELTA_POLARIZABILITY
    )

    MLJStarkModel = functools.partial(
        StarkModel, neutral_submodel=mlj_model, **stark_args
    )
    mlj_stark_model_dipole = MLJStarkModel(
        field_delta_dipole=DELTA_DIPOLE, field_delta_polarizability=0.0
    )
    mlj_stark_model_polarizability = MLJStarkModel(
        field_delta_dipole=0.0, field_delta_polarizability=DELTA_POLARIZABILITY
    )

    return tuple(
        model.get_absorption()
        for model in (
            two_state_model,
            two_state_stark_model_dipole,
            two_state_stark_model_polarizability,
            mlj_model,
            mlj_stark_model_dipole,
            mlj_stark_model_polarizability,
        )
    )


def get_spectra():
    if REGENERATE:
        outputs = compute_spectra()
        save_file(outputs, f"{PICKLE_PREFIX}/outputs.pkl")
    else:
        outputs = open_file(f"{PICKLE_PREFIX}/outputs.pkl")
    return outputs


def plot(axes, abs, stark_dipole, stark_polarizability):
    axes[0].plot(abs.energies, abs.intensities)

    abs_avg, *_ = get_stats(abs.energies, abs.intensities)
    axes[0].axvline(x=abs_avg, color="black", linestyle="--", label=r"$\bar{E}$")

    first_derivative = np.gradient(abs.intensities, abs.energies)
    second_derivative = np.gradient(first_derivative, abs.energies)

    for ax, stark in zip(axes[1:], [stark_dipole, stark_polarizability]):
        ax.plot(stark.energies, stark.intensities)

    stark_twins = [ax.twinx() for ax in axes[1:]]

    for ax, derivative in zip(stark_twins, [second_derivative, first_derivative]):
        ax.plot(abs.energies, derivative, color="orange")

    for ax, twin in zip(axes[1:], stark_twins):
        fix_twinx_ticks(ax, twin)


def main():
    (
        ts_abs,
        ts_stark_dipole,
        ts_stark_polar,
        mlj_abs,
        mlj_stark_dipole,
        mlj_stark_polar,
    ) = get_spectra()

    fig, axes = plt.subplots(3, 2)

    plot(axes[:, 0], ts_abs, ts_stark_dipole, ts_stark_polar)
    plot(axes[:, 1], mlj_abs, mlj_stark_dipole, mlj_stark_polar)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
