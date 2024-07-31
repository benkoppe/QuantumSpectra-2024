import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Int, Array

from quantumspectra_2024.common.absorption import AbsorptionModel as Model
from quantumspectra_2024.common.hamiltonian import HamiltonianModel


@jdc.pytree_dataclass(kw_only=True, eq=False, frozen=True)
class ThreeStateModel(Model):

    broadening: float = 200.0
    temperature_kelvin: float

    ct_energy_gap: float
    le_energy_gap: float

    gs_ct_coupling: float
    ct_le_coupling: float

    ct_mode_couplings: Float[Array, "num_modes"]
    le_mode_couplings: Float[Array, "num_modes"]

    mode_basis_sets: Int[Array, "num_modes"]
    mode_frequencies: Float[Array, "num_modes"]

    def get_hamiltonian(self) -> HamiltonianModel:
        return HamiltonianModel(
            transfer_integrals=self.gs_ct_coupling,
            state_energies=jnp.array([0.0, self.ct_energy_gap, self.le_energy_gap]),
            mode_basis_sets=jnp.array(self.mode_basis_sets),
            mode_localities=jnp.array([True, True, True]),
            mode_frequencies=jnp.array(self.mode_frequencies),
            mode_state_couplings=jnp.array(
                [
                    [0.0, ct_mode_coupling, le_mode_coupling]
                    for ct_mode_coupling, le_mode_coupling in zip(
                        self.ct_mode_couplings, self.le_mode_couplings
                    )
                ]
            ),
        )
