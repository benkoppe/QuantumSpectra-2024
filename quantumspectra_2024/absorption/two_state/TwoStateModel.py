import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Int, Array, Scalar

from quantumspectra_2024.modules.absorption import (
    AbsorptionModel as Model,
    AbsorptionSpectrum,
)
from quantumspectra_2024.modules.hamiltonian import HamiltonianModel
from quantumspectra_2024.absorption.two_state.TwoStateComputation import (
    compute_peaks,
    broaden_peaks,
)


@jdc.pytree_dataclass(kw_only=True)
class TwoStateModel(Model):
    """A two-state quantum mechanical model for absorption spectra."""

    temperature_kelvin: Float[Scalar, ""]
    broadening: Float[Scalar, ""]

    transfer_integral: Float[Scalar, ""]
    energy_gap: Float[Scalar, ""]

    mode_basis_sets: jdc.Static[Int[Array, "num_modes"]]
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

    def get_hamiltonian(self) -> HamiltonianModel:
        return HamiltonianModel(
            transfer_integral=self.transfer_integral,
            state_energies=jnp.array([0.0, self.energy_gap]),
            mode_basis_sets=jnp.array(self.mode_basis_sets),
            mode_localities=jnp.array([True, True]),
            mode_frequencies=jnp.array(self.mode_frequencies),
            mode_state_couplings=jnp.array(
                [[0.0, mode_coupling] for mode_coupling in self.mode_couplings]
            ),
        )

    def apply_electric_field(
        self,
        field_strength: Array,
        field_delta_dipole: Array,
        field_delta_polarizability: Array,
    ) -> "TwoStateModel":
        dipole_energy_change = field_delta_dipole * field_strength * 1679.0870295
        polarizability_energy_change = (
            0.5 * (field_strength**2) * field_delta_polarizability * 559.91
        )
        field_energy_change = -1 * (dipole_energy_change + polarizability_energy_change)

        return jdc.replace(
            self,
            energy_gap=self.energy_gap + field_energy_change,
        )
