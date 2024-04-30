import pickle
from pathlib import Path
from quantumspectra_2024.common.absorption import AbsorptionSpectrum

import jax

jax.config.update("jax_enable_x64", True)

PARENT_DIR = Path(__file__).parent


# PLOTTING

LINEWIDTH = 3

ENERGY_XLABEL = "E (eV)"

ABS_YLABEL = "A(E)"

# default font parameters
TITLEDICT = {"weight": "heavy", "size": 15}  # 15
LABELDICT = {"weight": "heavy", "size": 14}  # 14
LEGENDDICT = {"weight": "normal", "size": 13}  # 14


def make_bold(tex):
    return rf"$\mathbf{{{tex}}}$"


# DATA CONVERSION


def wavenumbers_to_ev(wavenumbers):
    if wavenumbers is None:
        return None
    return wavenumbers / 8065.54429


# PICKLING


def save_file(data, filename):
    with open(f"{PARENT_DIR}/{filename}", "wb") as f:
        pickle.dump(data, f)


def open_file(filename):
    with open(f"{PARENT_DIR}/{filename}", "rb") as f:
        return pickle.load(f)


# SPECTRUM OBJECT MUTATION


def match_spectrum_greatest_peak(base_spectrum: AbsorptionSpectrum, *other_spectra):
    greatest_peak_index = base_spectrum.intensities.argmax()
    greatest_peak_intensity = base_spectrum.intensities[greatest_peak_index]

    new_spectra = []
    for other_spectrum in other_spectra:
        other_greatest_peak_intensity = other_spectrum.intensities[
            other_spectrum.intensities.argmax()
        ]

        scaling_factor = greatest_peak_intensity / other_greatest_peak_intensity
        new_intensities = other_spectrum.intensities * scaling_factor

        new_spectrum = AbsorptionSpectrum(
            energies=other_spectrum.energies,
            intensities=new_intensities,
        )
        new_spectra.append(new_spectrum)

    if len(new_spectra) == 1:
        return new_spectra[0]
    return new_spectra


def scale_relative_to_one(spectra):
    greatest_peak_intensity = max([spectrum.intensities.max() for spectrum in spectra])

    new_spectra = []
    for spectrum in spectra:
        scaling_factor = 1 / greatest_peak_intensity
        new_intensities = spectrum.intensities * scaling_factor

        new_spectrum = AbsorptionSpectrum(
            energies=spectrum.energies,
            intensities=new_intensities,
        )
        new_spectra.append(new_spectrum)

    return new_spectra
