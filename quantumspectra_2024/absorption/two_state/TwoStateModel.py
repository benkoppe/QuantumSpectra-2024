import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Int, Array, Scalar

from quantumspectra_2024.modules.absorption import (
    AbsorptionModel as Model,
    AbsorptionSpectrum,
)
from quantumspectra_2024.absorption.two_state.TwoStateComputation import (
    compute_peaks,
    broaden_peaks,
)


@jdc.pytree_dataclass
class TwoStateModel(Model):
    """A two-state quantum mechanical model for absorption spectra."""

    temperature_kelvin: Float[Scalar, ""]
    broadening: Float[Scalar, ""]

    basis_sets: jdc.Static[Int[Array, "2"]]

    transfer_integral: Float[Scalar, ""]
    energy_gap: Float[Scalar, ""]

    mode_frequencies: Float[Array, "num_modes"]
    mode_couplings: Float[Array, "num_modes"]

    def get_absorption(self) -> AbsorptionSpectrum:
        """Compute the absorption spectrum for the model."""
        # compute the Hamiltonian
        hamiltonian = self.get_hamiltonian()

        # diagonalize the Hamiltonian
        eigenvalues, eigenvectors = hamiltonian.get_diagonalization()

        # get absorption spectrum sample energies (x values)
        sample_points = jnp.linspace(
            self.start_energy, self.end_energy, self.num_points
        )

        # compute absorption spectrum peaks
        peak_energies, peak_intensities = compute_peaks(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            transfer_integral=self.transfer_integral,
            temperature_kelvin=self.temperature_kelvin,
        )

        # broaden peaks into spectrum
        spectrum = broaden_peaks(
            sample_points=sample_points,
            peak_energies=peak_energies,
            peak_intensities=peak_intensities,
            distribution_broadening=self.broadening,
        )

        # return as AbsorptionSpectrum dataclass
        return AbsorptionSpectrum(
            energies=sample_points,
            intensities=spectrum,
        )
