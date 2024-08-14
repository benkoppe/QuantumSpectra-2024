from quantumspectra_2024.models import TwoStateModel, MLJModel, StarkModel

import numpy as np
import matplotlib.pyplot as plt


def normalize_max_positive_y_peak(
    base_y: np.ndarray, other_y: np.ndarray
) -> np.ndarray:
    base_max = np.max(base_y)
    other_max = np.max(other_y)
    return other_y * base_max / other_max


def ev_to_wavenumbers(ev: float) -> float:
    return ev * 8065.54429


two_stark_model = StarkModel(
    neutral_submodel=TwoStateModel(
        temperature_kelvin=300,
        energy_gap=ev_to_wavenumbers(0.5),
        transfer_integral=100,
        mode_basis_sets=[20, 200],
        mode_frequencies=[1200, 100],
        mode_couplings=[2.0, 4.0],
    ),
    positive_field_strength=0.1,
    positive_field_sum_percent=0.5,
    field_delta_dipole=10,
    field_delta_polarizability=100,
)

mlj_stark_model = StarkModel(
    neutral_submodel=MLJModel(
        temperature_kelvin=two_stark_model.neutral_submodel.temperature_kelvin,
        energy_gap=two_stark_model.neutral_submodel.energy_gap,
        disorder_meV=0,
        basis_size=20,
        mode_frequencies=two_stark_model.neutral_submodel.mode_frequencies,
        mode_couplings=two_stark_model.neutral_submodel.mode_couplings,
    ),
    positive_field_strength=two_stark_model.positive_field_strength,
    positive_field_sum_percent=two_stark_model.positive_field_sum_percent,
    field_delta_dipole=two_stark_model.field_delta_dipole,
    field_delta_polarizability=two_stark_model.field_delta_polarizability,
)

two_abs = two_stark_model.get_absorption()
mlj_abs = mlj_stark_model.get_absorption()

mlj_abs_adjusted_intensity = normalize_max_positive_y_peak(
    two_abs.intensities, mlj_abs.intensities
)

from dataclasses import asdict
from pathlib import Path

parent = Path(__file__).parent


def dict_to_str(dict, indent=0):
    combiner = "\n" + "\t" * indent
    for key in ["start_energy", "end_energy", "num_points"]:
        dict.pop(key, None)
    return combiner + combiner.join([f"{k}: {v}" for k, v in dict.items()])


def write_args(model, filename):
    with open(parent / filename, "w") as f:
        submodel_dict = asdict(model.neutral_submodel)
        dict = asdict(model)

        dict["neutral_submodel"] = dict_to_str(submodel_dict, 1)
        f.write(dict_to_str(dict))


write_args(two_stark_model, "two-stark-args.txt")
write_args(mlj_stark_model, "mlj-stark-args.txt")

plt.plot(two_abs.energies, two_abs.intensities, label="two-state")
plt.plot(mlj_abs.energies, mlj_abs_adjusted_intensity, label="mlj")
plt.title("Two-state vs MLJ Stark model absorption")
plt.xlabel("Energy (cm^-1)")
plt.ylabel("Intensity")
plt.legend()
plt.show()
