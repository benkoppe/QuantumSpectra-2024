import pickle
from pathlib import Path
from quantumspectra_2024.common.absorption import AbsorptionSpectrum

import jax
import numpy as np

# jax.config.update("jax_enable_x64", True)

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


# STATISTICAL ANALYSIS


def get_stats(x, y):
    # Returns a distribution's mean, standard deviation, skewness, and kurtosis

    x, y = np.array(x), np.array(y)
    y = np.abs(y)

    # Calculate the mean and standard deviation
    mean = np.sum(x * y) / np.sum(y)
    variance = np.sum(y * (x - mean) ** 2) / np.sum(y)
    std_dev = np.sqrt(variance)

    # Calculate the third and fourth central moments of the data.
    skewness = np.sum(y * (x - mean) ** 3) / (np.sum(y) * std_dev**3)
    kurtosis = np.sum(y * (x - mean) ** 4) / (np.sum(y) * std_dev**4) - 3

    return mean, std_dev, skewness, kurtosis


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


# PLOT HELPERS


def fix_twinx_ticks(ax, twinax):
    y1_min, y1_max = ax.get_ylim()
    y2_min, y2_max = twinax.get_ylim()

    ratio1 = y1_max / (-y1_min if y1_min != 0 else y1_max)
    ratio2 = y2_max / (-y2_min if y2_min != 0 else y2_max)

    ax.set_zorder(1)
    twinax.set_zorder(0)
    ax.patch.set_visible(False)

    # Set the same ratio for both axes
    if ratio1 > ratio2:
        new_y2_max = -y2_min * ratio1
        twinax.set_ylim(y2_min, new_y2_max)
    else:
        new_y1_max = -y1_min * ratio2
        ax.set_ylim(y1_min, new_y1_max)
