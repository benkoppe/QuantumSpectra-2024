from pathlib import Path
import os
import pickle

from quantumspectra_2024.models import TwoStateModel, StarkModel
from quantumspectra_2024.common.absorption import AbsorptionSpectrum

PARENT_DIR = Path(__file__).parent
PICKLE_DIR = PARENT_DIR / "pickles"


def get_filename(model) -> Path:
    model_type = model.__class__.__name__
    model_dir = PICKLE_DIR / model_type

    model_str = model.__str__()
    model_str = str(hash(model_str))

    model_str += ".pkl"

    return model_dir / model_str


def save_spectrum(model, spectrum) -> Path:
    fname = get_filename(model)

    # ensure directory exists
    os.makedirs(fname.parent, exist_ok=True)

    with open(fname, "wb") as f:
        pickle.dump(spectrum, f)

    return fname


def load_spectrum(model) -> AbsorptionSpectrum:
    fname = get_filename(model)

    # load the spectrum
    with open(fname, "rb") as f:
        spectrum = pickle.load(f)

    return spectrum


def compute_or_load_spectrum(model) -> AbsorptionSpectrum:
    try:
        return load_spectrum(model)
    except FileNotFoundError:
        spectrum = model.get_absorption()
        save_spectrum(model, spectrum)
        return spectrum


model = TwoStateModel(
    temperature_kelvin=300,
    transfer_integral=100,
    energy_gap=8_000,
    mode_basis_sets=[20, 200],
    mode_frequencies=[1200, 100],
    mode_couplings=[0.7, 2.0],
)

smodel = StarkModel(
    neutral_submodel=model,
    positive_field_strength=0.01,
    field_delta_dipole=0,
    field_delta_polarizability=0,
)

print(get_filename(smodel))
