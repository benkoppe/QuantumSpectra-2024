import jax_dataclasses as jdc
from jaxtyping import Float, Int, Bool, Array

from quantumspectra_2024.common.hamiltonian.HamiltonianComputation import (
    build_matrix,
    diagonalize_matrix,
)


@jdc.pytree_dataclass(kw_only=True)
class HamiltonianModel:
    """A hamiltonian model for a quantum system.

    Attributes
    ----------
    transfer_integral : float
        transfer integral between states.
    state_energies : Float[Array, "num_states"]
        energies of each state.

    mode_basis_sets : Int[Array, "num_modes"]
        basis set size per mode.
    mode_localities : Bool[Array, "num_modes"]
        whether each mode is local.
    mode_frequencies : Float[Array, "num_modes"]
        frequency per mode.
    mode_state_couplings : Float[Array, "num_modes num_states"]
        coupling per mode and state.
    """

    #: transfer integral between states.
    transfer_integral: float
    #: energies of each state.
    state_energies: Float[Array, "num_states"]

    #: basis set size per mode.
    mode_basis_sets: Int[Array, "num_modes"]
    #: whether each mode is local.
    mode_localities: Bool[Array, "num_modes"]
    #: frequency per mode.
    mode_frequencies: Float[Array, "num_modes"]
    #: coupling per mode and state.
    mode_state_couplings: Float[Array, "num_modes num_states"]

    def get_diagonalization(self) -> tuple[
        Float[Array, "matrix_size"],
        Float[Array, "matrix_size matrix_size"],
    ]:
        """Compute the diagonalization of the Hamiltonian.

        First computes the Hamiltonian matrix, then diagonalizes it.
        See docs in `HamiltonianComputation` to see how this is done.

        Returns
        -------
        tuple[Float[Array, "matrix_size"], Float[Array, "matrix_size matrix_size"]]
            tuple containing the eigenvalues and eigenvectors in `jax` arrays.
        """
        # build matrix
        matrix = self.get_matrix()

        # diagonalize matrix
        eigenvalues, eigenvectors = diagonalize_matrix(matrix)

        return eigenvalues, eigenvectors

    def get_matrix(self) -> Float[Array, "matrix_size matrix_size"]:
        """Builds the Hamiltonian matrix.

        See docs in `HamiltonianComputation` to see how this is done.

        Returns
        -------
        Float[Array, "matrix_size matrix_size"]
            fully constructed matrix in `jax` array.
        """
        return build_matrix(
            mode_basis_sets=self.mode_basis_sets,
            state_energies=self.state_energies,
            transfer_integral=self.transfer_integral,
            mode_localities=self.mode_localities,
            mode_frequencies=self.mode_frequencies,
            mode_state_couplings=self.mode_state_couplings,
        )
