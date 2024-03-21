import jax_dataclasses as jdc
from jaxtyping import Float, Int, Bool, Scalar, Array

from quantumspectra_2024.modules.hamiltonian.HamiltonianComputation import (
    build_matrix,
    diagonalize_matrix,
)


@jdc.pytree_dataclass
class HamiltonianModel:
    """A hamiltonian model for a quantum system"""

    basis_sets: jdc.Static[Int[Array, "num_modes"]]

    transfer_integral: Float[Scalar, ""]
    state_energies: Float[Array, "num_states"]

    mode_localities: Bool[Array, "num_modes"]
    mode_frequencies: Float[Array, "num_modes"]
    state_mode_couplings: Float[Array, "num_states num_modes"]

    def get_diagonalization(self) -> tuple[
        Float[Array, "num_states*block_size"],
        Float[Array, "num_states*block_size num_states*block_size"],
    ]:
        # build matrix
        matrix = build_matrix(
            basis_sets=self.basis_sets,
            state_energies=self.state_energies,
            transfer_integral=self.transfer_integral,
            mode_localities=self.mode_localities,
            mode_frequencies=self.mode_frequencies,
            state_mode_couplings=self.state_mode_couplings,
        )

        # diagonalize matrix
        eigenvalues, eigenvectors = diagonalize_matrix(matrix)

        return eigenvalues, eigenvectors
