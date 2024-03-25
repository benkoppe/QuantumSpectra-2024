import math

import numpy as np
from jaxtyping import Float, Int, Array, Scalar


def calculate_mlj_spectrum(
    energy_gap: Float[Scalar, ""],
    high_freq_frequency: Float[Scalar, ""],
    high_freq_coupling: Float[Scalar, ""],
    low_freq_frequency: Float[Scalar, ""],
    low_freq_coupling: Float[Scalar, ""],
    temperature_kelvin: Float[Scalar, ""],
    disorder_meV: Float[Scalar, ""],
    basis_size: Int[Scalar, ""],
    sample_points: Float[Array, "num_points"],
) -> Float[Array, "num_points"]:
    # compute necessary values
    disorder_wavenumbers = disorder_meV * 8061 * 0.001
    low_freq_relaxation_energy = low_freq_coupling**2 * low_freq_frequency
    temperature_kbT = temperature_kelvin * 0.695028
    high_freq_huang_rhys_factor = high_freq_coupling**2

    factorials = [math.factorial(n) for n in range(basis_size + 1)]

    def calculate_mlj_single_intensity(
        energy: Float[Scalar, ""],
    ) -> Float[Scalar, ""]:
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
