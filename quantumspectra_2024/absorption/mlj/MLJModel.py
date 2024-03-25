import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Int, Array, Scalar

from quantumspectra_2024.modules.absorption import (
    AbsorptionModel as Model,
    AbsorptionSpectrum,
)
from quantumspectra_2024.absorption.mlj.MLJComputation import calculate_mlj_spectrum


@jdc.pytree_dataclass(kw_only=True)
class MLJModel(Model):
    """A two-state two-mode MLJ model for absorption spectra."""

    temperature_kelvin: Float[Scalar, ""]
    energy_gap: Float[Scalar, ""]
    disorder_meV: Float[Scalar, ""]

    basis_size: jdc.Static[Int[Scalar, ""]]

    mode_frequencies: Float[Array, "2"]
    mode_couplings: Float[Array, "2"]

    def get_absorption(self) -> AbsorptionSpectrum:
        # get absorption spectrum sample energies (x values)
        sample_points = jnp.linspace(
            float(self.start_energy), float(self.end_energy), int(self.num_points)
        )

        # get low and high frequency modes
        lower_frequency, lower_coupling, higher_frequency, higher_coupling = (
            self.get_low_high_frequency_modes()
        )

        # calculate absorption spectrum
        spectrum = calculate_mlj_spectrum(
            energy_gap=float(self.energy_gap),
            high_freq_frequency=float(higher_frequency),
            high_freq_coupling=float(higher_coupling),
            low_freq_frequency=float(lower_frequency),
            low_freq_coupling=float(lower_coupling),
            temperature_kelvin=float(self.temperature_kelvin),
            disorder_meV=float(self.disorder_meV),
            basis_size=int(self.basis_size),
            sample_points=np.array(sample_points),
        )

        # return as AbsorptionSpectrum dataclass
        return AbsorptionSpectrum(
            energies=sample_points,
            intensities=jnp.array(spectrum),
        )

    def get_low_high_frequency_modes(self):
        sorted_frequency_indices = np.argsort(self.mode_frequencies)

        sorted_frequencies = self.mode_frequencies[sorted_frequency_indices]
        sorted_couplings = self.mode_couplings[sorted_frequency_indices]

        # Assign to variables
        lower_frequency = sorted_frequencies[0]
        lower_coupling = sorted_couplings[0]
        higher_frequency = sorted_frequencies[-1]
        higher_coupling = sorted_couplings[-1]

        return lower_frequency, lower_coupling, higher_frequency, higher_coupling

    def verify_modes(self):
        if len(self.mode_frequencies) != 2 or len(self.mode_couplings) != 2:
            raise ValueError("The MLJ model requires exactly two modes.")

    def __post_init__(self):
        self.verify_modes()
        super().__post_init__()
