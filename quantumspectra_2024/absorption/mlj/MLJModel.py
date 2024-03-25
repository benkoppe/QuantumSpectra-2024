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
    """A two-state two-mode MLJ model for absorption spectra.

    Attributes:
        start_energy (Float[Scalar, ""]): absorption spectrum's starting energy.
        end_energy (Float[Scalar, ""]): absorption spectrum's ending energy.
        num_points (Int[Scalar, ""]): absorption spectrum's number of points.

        temperature_kelvin (Float[Scalar, ""]): system's temperature in Kelvin.
        energy_gap (Float[Scalar, ""]): energy gap between the two states.
        disorder_meV (Float[Scalar, ""]): disorder in the system in meV.

        basis_size (Int[Scalar, ""]): size of basis set.

        mode_frequencies (Float[Array, "2"]): frequency per mode.
        mode_couplings (Float[Array, "2"]): excited state coupling per mode.
    """

    temperature_kelvin: Float[Scalar, ""]
    energy_gap: Float[Scalar, ""]
    disorder_meV: Float[Scalar, ""]

    basis_size: jdc.Static[Int[Scalar, ""]] = 20

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
        mode_frequencies = np.array(self.mode_frequencies)
        mode_couplings = np.array(self.mode_couplings)

        sorted_frequency_indices = np.argsort(mode_frequencies)

        sorted_frequencies = mode_frequencies[sorted_frequency_indices]
        sorted_couplings = mode_couplings[sorted_frequency_indices]

        # Assign to variables
        lower_frequency = sorted_frequencies[0]
        lower_coupling = sorted_couplings[0]
        higher_frequency = sorted_frequencies[-1]
        higher_coupling = sorted_couplings[-1]

        return lower_frequency, lower_coupling, higher_frequency, higher_coupling

    def apply_electric_field(
        self,
        field_strength: Array,
        field_delta_dipole: Array,
        field_delta_polarizability: Array,
    ) -> "MLJModel":
        dipole_energy_change = field_delta_dipole * field_strength * 1679.0870295
        polarizability_energy_change = (
            0.5 * (field_strength**2) * field_delta_polarizability * 559.91
        )
        field_energy_change = -1 * (dipole_energy_change + polarizability_energy_change)

        return jdc.replace(
            self,
            energy_gap=self.energy_gap + field_energy_change,
        )

    def verify_modes(self):
        if len(self.mode_frequencies) != 2 or len(self.mode_couplings) != 2:
            raise ValueError("The MLJ model requires exactly two modes.")

    def __post_init__(self):
        self.verify_modes()
