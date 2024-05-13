import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Int, Array

from quantumspectra_2024.common.absorption import (
    AbsorptionModel as Model,
    AbsorptionSpectrum,
)
from quantumspectra_2024.common.hamiltonian import HamiltonianModel
from quantumspectra_2024.common.hamiltonian.HamiltonianComputation import (
    calculate_state_local_diagonals,
)
from quantumspectra_2024.models.two_state.TwoStateComputation import (
    broaden_peaks,
)
from quantumspectra_2024.models.three_state.ThreeStateComputation import (
    compute_peaks,
)


@jdc.pytree_dataclass(kw_only=True, eq=True, frozen=True)
class ThreeStateSimpleModel(Model):

    broadening: float = 200.0
    temperature_kelvin: float

    ct_energy_gap: float
    le_energy_gap: float

    gs_ct_coupling: float
    ct_le_coupling: float

    d_LE: float  # TODO: dipole moment for LE state?
    d_CT: float  # dipole moment for CT state?

    ct_mode_couplings: Float[Array, "num_modes"]
    le_mode_couplings: Float[Array, "num_modes"]

    mode_basis_sets: Int[Array, "num_modes"]
    mode_frequencies: Float[Array, "num_modes"]

    def get_absorption(self) -> AbsorptionSpectrum:
        # get absorption spectrum sample energies (x values)
        sample_points = jnp.linspace(
            self.start_energy, self.end_energy, self.num_points
        )

        # get two-state hamiltonian
        two_state_hamiltonian = self.get_hamiltonian()

        # get diagonal of cheated hamiltonian
        two_state_eigenvalues, two_state_eigenvectors = (
            two_state_hamiltonian.get_diagonalization()
        )

        # get peaks
        peak_energies, peak_intensities = compute_peaks(
            two_state_eigenvalues=two_state_eigenvalues,
            two_state_eigenvectors=two_state_eigenvectors,
            d_LE=self.d_LE,
            d_CT=self.d_CT,
            temperature_kelvin=self.temperature_kelvin,
        )

        # broaden peaks into spectrum
        spectrum = broaden_peaks(
            sample_points=sample_points,
            peak_energies=peak_energies,
            peak_intensities=peak_intensities,
            distribution_broadening=self.broadening,
        )

        return AbsorptionSpectrum(
            energies=sample_points,
            intensities=spectrum,
        )

    def get_hamiltonian(self) -> HamiltonianModel:
        return HamiltonianModel(
            transfer_integral=self.ct_le_coupling,
            state_energies=jnp.array([self.le_energy_gap, self.ct_energy_gap]),
            mode_basis_sets=jnp.array(self.mode_basis_sets),
            mode_localities=jnp.array([True, True]),
            mode_frequencies=jnp.array(self.mode_frequencies),
            mode_state_couplings=jnp.array(
                [
                    [le_mode_coupling, ct_mode_coupling]
                    for le_mode_coupling, ct_mode_coupling in zip(
                        self.le_mode_couplings, self.ct_mode_couplings
                    )
                ]
            ),
        )

    def apply_electric_field(
        field_strength: float,
        field_delta_dipole: float,
        field_delta_polarizability: float,
    ) -> Model:
        return super().apply_electric_field(
            field_delta_dipole, field_delta_polarizability
        )
