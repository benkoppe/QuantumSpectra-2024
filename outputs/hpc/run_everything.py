from pathlib import Path
from dataclasses import dataclass
from itertools import product

import numpy as np

from quantumspectra_2024.models import StarkModel, TwoStateModel, MLJModel
from saving_loading import compute_or_load_spectrum

PARENT_DIR = Path(__file__).parent
PICKLE_DIR = PARENT_DIR / "pickles"

# STATIC ARGUMENTS

COMMON_ARGS = {
    "temperature_kelvin": 300,
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
}

# VARIABLE ARGUMENTS


@dataclass
class VariableArg:
    values: np.ndarray

    def __init__(self, values):
        self.values = np.array(values)

    @staticmethod
    def from_minmax(min, max, num_points):
        return VariableArg(np.linspace(min, max, num_points))


VARIABLE_ARGS = {
    "energy_gap": VariableArg.from_minmax(0, 8_000, 10),
    "g1": VariableArg([0.7]),
    "g2": VariableArg.from_minmax(2, 8, 4),
    "positive_field_strength": VariableArg.from_minmax(0.01, 0.1, 4),
    "field_delta_dipole": VariableArg.from_minmax(0, 38, 4),
    "field_delta_polarizability": VariableArg.from_minmax(0, 1_000, 4),
}


# Computing a spectrum

TWO_STATE_VALID_ARGS = TwoStateModel.get_arguments()
MLJ_VALID_ARGS = MLJModel.get_arguments()
STARK_VALID_ARGS = StarkModel.get_arguments()


def compute_spectra(variable_args):
    all_args = {
        **COMMON_ARGS,
        **variable_args,
        **TWO_STATE_ARGS,
        **STARK_ARGS,
        **MLJ_ARGS,
    }

    two_state_args = {k: v for k, v in all_args.items() if k in TWO_STATE_VALID_ARGS}
    mlj_args = {k: v for k, v in all_args.items() if k in MLJ_VALID_ARGS}
    stark_args = {k: v for k, v in all_args.items() if k in STARK_VALID_ARGS}

    mode_coupings = [variable_args["g1"], variable_args["g2"]]
    two_state_args["mode_couplings"] = mode_coupings
    mlj_args["mode_couplings"] = mode_coupings

    two_state_model = TwoStateModel(**two_state_args)
    mlj_model = MLJModel(**mlj_args)

    two_state_stark_model = StarkModel(neutral_submodel=two_state_model, **stark_args)
    mlj_stark_model = StarkModel(neutral_submodel=mlj_model, **stark_args)

    return (
        compute_or_load_spectrum(two_state_model),
        compute_or_load_spectrum(mlj_model),
        compute_or_load_spectrum(two_state_stark_model),
        compute_or_load_spectrum(mlj_stark_model),
    )


def main():
    param_values = list(product(*[arg.values for arg in VARIABLE_ARGS.values()]))
    labeled_param_values = [
        dict(zip(VARIABLE_ARGS.keys(), values)) for values in param_values
    ]

    for i, param_values in enumerate(labeled_param_values):
        print(f"Computing spectrum {i + 1}/{len(labeled_param_values)}")
        compute_spectra(param_values)


if __name__ == "__main__":
    main()
