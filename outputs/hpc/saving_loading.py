from pathlib import Path
import os
import pickle
from abc import ABC

import numpy as np

from quantumspectra_2024.models import TwoStateModel, StarkModel
from quantumspectra_2024.common.absorption import AbsorptionSpectrum, AbsorptionModel

PARENT_DIR = Path(__file__).parent
PICKLE_DIR = PARENT_DIR / "pickles"
CSV_DIR = PARENT_DIR / "csvs"


class SAVELOAD(ABC):
    extension: str
    save_dir: Path

    @classmethod
    def save(cls, fname: str, spectrum: AbsorptionSpectrum):
        pass

    @classmethod
    def load(cls, fname: str) -> AbsorptionSpectrum:
        pass


class PICKLE(SAVELOAD):
    extension = ".pkl"
    save_dir = PICKLE_DIR

    @classmethod
    def save(cls, fname: str, spectrum: AbsorptionSpectrum):
        with open(fname, "wb") as f:
            pickle.dump(spectrum, f)

    @classmethod
    def load(cls, fname: str) -> AbsorptionSpectrum:
        with open(fname, "rb") as f:
            return pickle.load(f)


class CSV(SAVELOAD):
    extension = ".csv"
    save_dir = CSV_DIR

    @classmethod
    def save(cls, fname: str, spectrum: AbsorptionSpectrum):
        spectrum.save_data(fname)

    @classmethod
    def load(cls, fname: str) -> AbsorptionSpectrum:
        energies, intensities = np.loadtxt(fname, delimiter=",", unpack=True)
        return AbsorptionSpectrum(energies, intensities)


DEFAULT_SAVELOAD = CSV


def get_filename(model: AbsorptionModel, save_dir=PICKLE_DIR, extension=".pkl") -> Path:
    model_type = model.__class__.__name__
    model_dir = save_dir / model_type

    model_str = str(hash(model))

    model_str += extension

    return model_dir / model_str


def save_spectrum(
    model: AbsorptionModel,
    spectrum: AbsorptionSpectrum,
    save_type: SAVELOAD = DEFAULT_SAVELOAD,
) -> Path:
    fname = get_filename(
        model, save_dir=save_type.save_dir, extension=save_type.extension
    )

    # ensure directory exists
    os.makedirs(fname.parent, exist_ok=True)

    save_type.save(fname, spectrum)

    return fname


def load_spectrum(
    model: AbsorptionModel, save_type: SAVELOAD = DEFAULT_SAVELOAD
) -> AbsorptionSpectrum:
    fname = get_filename(
        model, save_dir=save_type.save_dir, extension=save_type.extension
    )

    # load the spectrum
    spectrum: AbsorptionSpectrum = save_type.load(fname)

    return spectrum


def compute_or_load_spectrum(
    model: AbsorptionModel, save_type: SAVELOAD = DEFAULT_SAVELOAD
) -> AbsorptionSpectrum:
    try:
        return load_spectrum(model, save_type=save_type)
    except FileNotFoundError:
        spectrum = model.get_absorption()
        save_spectrum(model, spectrum, save_type=save_type)
        return spectrum


def is_spectrum_saved(
    model: AbsorptionModel, save_type: SAVELOAD = DEFAULT_SAVELOAD
) -> bool:
    fname = get_filename(
        model, save_dir=save_type.save_dir, extension=save_type.extension
    )
    return fname.exists()
