from typing import Literal
from dataclasses import dataclass
from itertools import product

import numpy as np

from quantumspectra_2024.models import StarkModel, TwoStateModel, MLJModel
from outputs.hpc.saving_loading import (
    load_spectrum,
    compute_or_load_spectrum,
    is_spectrum_saved,
)

# STATIC ARGUMENTS

COMMON_ARGS = {
    "temperature_kelvin": 300,
    "mode_frequencies": [1700, 100],
    # "mode_couplings": [1.5, 6.67],
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

    @classmethod
    def from_minmax(cls, min, max, num_points):
        return cls(np.linspace(min, max, num_points))


VARIABLE_ARGS = {
    "energy_gap": VariableArg([0, 200, 400, 800, 8_000]),
    "g1": VariableArg([0.7]),
    "g2": VariableArg.from_minmax(2, 8, 4),
    "positive_field_strength": VariableArg([0.0, 0.01, 0.1]),
    "field_delta_dipole": VariableArg.from_minmax(0, 38, 3),
    "field_delta_polarizability": VariableArg.from_minmax(0, 1_000, 3),
}


# Computing a spectrum

TWO_STATE_VALID_ARGS = TwoStateModel.get_arguments()
MLJ_VALID_ARGS = MLJModel.get_arguments()
STARK_VALID_ARGS = StarkModel.get_arguments()


def get_models(all_args_dict, model: Literal["two_state", "mlj", "both"] = "both"):
    two_state_args = {
        k: v for k, v in all_args_dict.items() if k in TWO_STATE_VALID_ARGS
    }
    mlj_args = {k: v for k, v in all_args_dict.items() if k in MLJ_VALID_ARGS}
    stark_args = {k: v for k, v in all_args_dict.items() if k in STARK_VALID_ARGS}

    two_state_model = TwoStateModel(**two_state_args)
    mlj_model = MLJModel(**mlj_args)

    two_state_stark_model = StarkModel(neutral_submodel=two_state_model, **stark_args)
    mlj_stark_model = StarkModel(neutral_submodel=mlj_model, **stark_args)

    models = []

    if model == "two_state" or model == "both":
        models.append(two_state_model)
    if model == "mlj" or model == "both":
        models.append(mlj_model)
    if model == "two_state" or model == "both":
        models.append(two_state_stark_model)
    if model == "mlj" or model == "both":
        models.append(mlj_stark_model)

    return models


def compute_spectra(
    all_args_dict,
    skip_loading=False,
    force_loading=False,
    model: Literal["two_state", "mlj", "both"] = "both",
):
    if skip_loading and force_loading:
        raise ValueError("skip_loading and force_loading cannot both be True.")

    models = get_models(all_args_dict, model=model)

    # check for whether all models are computed
    if all(is_spectrum_saved(model) for model in models):
        # if skip_loading is True, return None
        if skip_loading:
            return None
        return tuple(load_spectrum(model) for model in models)

    # if force_loading is True, throw an error
    if force_loading:
        raise ValueError("force_loading is True, but not all models are computed.")

    return tuple(compute_or_load_spectrum(model) for model in models)


def main():
    param_values = list(product(*[arg.values for arg in VARIABLE_ARGS.values()]))
    labeled_param_values = [
        dict(zip(VARIABLE_ARGS.keys(), values)) for values in param_values
    ]

    for i, param_values in enumerate(labeled_param_values):
        print(f"{i + 1}/{len(labeled_param_values)}")

        all_args = {
            **COMMON_ARGS,
            **TWO_STATE_ARGS,
            **MLJ_ARGS,
            **STARK_ARGS,
            **param_values,
        }

        if "g1" and "g2" in all_args:
            g1 = all_args.pop("g1")
            g2 = all_args.pop("g2")
            all_args["mode_couplings"] = [g1, g2]

        compute_spectra(all_args, skip_loading=True)


if __name__ == "__main__":
    main()
