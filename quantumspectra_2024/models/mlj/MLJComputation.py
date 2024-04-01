import math

import numpy as np
from jaxtyping import Float, Array


def calculate_mlj_spectrum(
    energy_gap: float,
    high_freq_frequency: float,
    high_freq_coupling: float,
    low_freq_frequency: float,
    low_freq_coupling: float,
    temperature_kelvin: float,
    disorder_meV: float,
    basis_size: int,
    sample_points: Float[Array, "num_points"],
) -> Float[Array, "num_points"]:
    """Computes an MLJ semiclassical absorption spectrum.

    Notes
    -----
    TODO:
    * Add more detailed implemention description.

    Parameters
    ----------
    energy_gap : float
        energy gap between states.
    high_freq_frequency : float
        frequency of high frequency mode.
    high_freq_coupling : float
        coupling of high frequency mode.
    low_freq_frequency : float
        frequency of low frequency mode.
    low_freq_coupling : float
        coupling of low frequency mode.
    temperature_kelvin : float
        temperature in Kelvin.
    disorder_meV : float
        disorder (sigma) in meV.
    basis_size : int
        size of basis set.
    sample_points : Float[Array, "num_points"]
        energy points to sample spectrum.

    Returns
    -------
    Float[Array, "num_points"]
        Computed absorption spectrum.
    """
    # compute necessary values
    disorder_wavenumbers = disorder_meV * 8061 * 0.001
    low_freq_relaxation_energy = low_freq_coupling**2 * low_freq_frequency
    temperature_kbT = temperature_kelvin * 0.695028
    high_freq_huang_rhys_factor = high_freq_coupling**2

    factorials = [math.factorial(n) for n in range(basis_size + 1)]

    def calculate_mlj_single_intensity(energy: float) -> float:
        abs_arr = [
            (
                np.exp(-high_freq_huang_rhys_factor)
                * (high_freq_huang_rhys_factor**n)
                / factorials[n]
            )
            * np.exp(
                -(
                    (
                        n * high_freq_frequency
                        + energy_gap
                        + low_freq_relaxation_energy
                        - energy
                    )
                    ** 2
                )
                / (
                    4 * low_freq_relaxation_energy * temperature_kbT
                    + 2 * disorder_wavenumbers**2
                )
            )
            for n in range(basis_size + 1)
        ]
        return np.sum(abs_arr)

    spectrum = np.vectorize(calculate_mlj_single_intensity)(sample_points)
    spectrum = spectrum / np.max(spectrum)
    return spectrum
