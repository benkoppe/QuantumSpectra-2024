import jax_dataclasses as jdc
from jaxtyping import Float, Int, Bool, Scalar, Array

from quantumspectra_2024.modules.hamiltonian.HamiltonianComputation import (
    build_matrix,
    diagonalize_matrix,
)


@jdc.pytree_dataclass(kw_only=True)
class HamiltonianModel:
    """A hamiltonian model for a quantum system"""

    transfer_integral: Float[Scalar, ""]
    state_energies: Float[Array, "num_states"]

    mode_basis_sets: jdc.Static[Int[Array, "num_modes"]]
    mode_localities: Bool[Array, "num_modes"]
    mode_frequencies: Float[Array, "num_modes"]
    state_mode_couplings: Float[Array, "num_modes num_states"]

    def get_diagonalization(self) -> tuple[
        Float[Array, "matrix_size"],
        Float[Array, "matrix_size matrix_size"],
    ]:
        # build matrix
        matrix = self.get_matrix()

        # diagonalize matrix
        eigenvalues, eigenvectors = diagonalize_matrix(matrix)

        return eigenvalues, eigenvectors

    def get_matrix(self) -> Float[Array, "matrix_size matrix_size"]:
        return build_matrix(
            mode_basis_sets=self.mode_basis_sets,
            state_energies=self.state_energies,
            transfer_integral=self.transfer_integral,
            mode_localities=self.mode_localities,
            mode_frequencies=self.mode_frequencies,
            state_mode_couplings=self.state_mode_couplings,
        )
