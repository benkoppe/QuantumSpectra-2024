import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Bool, Array, Scalar

import itertools


def diagonalize_matrix(matrix: Float[Array, "matrix_size matrix_size"]) -> tuple[
    Float[Array, "matrix_size"],
    Float[Array, "matrix_size matrix_size"],
]:
    """Diagonalizes a matrix and returns the eigenvalues and eigenvectors.

    Args:
        matrix (Float[Array, "matrix_size matrix_size"]): a matrix to diagonalize

    Returns:
        tuple[Float[Array, "matrix_size"], Float[Array, "matrix_size matrix_size"]]: a tuple containing the eigenvalues and eigenvectors
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def build_matrix(
    state_energies: Float[Array, "num_states"],
    transfer_integral: Float[Scalar, ""],
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_state_couplings: Float[Array, "num_modes num_states"],
) -> Float[Array, "matrix_size matrix_size"]:
    num_states = len(state_energies)

    # build the matrix, state by state
    rows = []
    for state_row in range(num_states):
        cols = []
        for state_col in range(num_states):
            state_index = max(state_row, state_col)

            if state_row == state_col:
                # calculate a local state block
                state = build_local_state_block(
                    state_index=state_index,
                    state_energies=state_energies,
                    mode_basis_sets=mode_basis_sets,
                    mode_localities=mode_localities,
                    mode_frequencies=mode_frequencies,
                    mode_state_couplings=mode_state_couplings,
                )
            else:
                # calculate a nonlocal state block
                state = build_nonlocal_state_block(
                    state_index=state_index,
                    transfer_integral=transfer_integral,
                    mode_basis_sets=mode_basis_sets,
                    mode_localities=mode_localities,
                    mode_frequencies=mode_frequencies,
                    mode_state_couplings=mode_state_couplings,
                )
            cols.append(state)
        rows.append(jnp.hstack(cols))
    matrix = jnp.vstack(rows)

    return matrix


def build_local_state_block(
    state_index: Int[Scalar, ""],
    state_energies: Float[Array, "num_states"],
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_state_couplings: Float[Array, "num_modes num_states"],
) -> Float[Array, "block_size block_size"]:
    state_energy = state_energies[state_index]
    mode_couplings = mode_state_couplings[:, state_index]

    # calculate the state's diagonal values
    all_diagonal_values = calculate_state_local_diagonals(
        state_energy=state_energy,
        mode_frequencies=mode_frequencies,
        mode_couplings=mode_couplings,
        mode_basis_sets=mode_basis_sets,
    )

    # calculate the state's offdiagonal values, arranged in a tuple for each mode
    all_mode_offdiagonal_values = calculate_state_offdiagonals(
        state_locality=True,
        mode_basis_sets=mode_basis_sets,
        mode_localities=mode_localities,
        mode_frequencies=mode_frequencies,
        mode_couplings=mode_couplings,
    )

    # build a state block with the diagonal values and tuple of offdiagonal values
    return build_state_block(
        all_diagonal_values=all_diagonal_values,
        all_mode_offdiagonal_values=all_mode_offdiagonal_values,
        mode_basis_sets=mode_basis_sets,
    )


def build_nonlocal_state_block(
    state_index: Int[Scalar, ""],
    transfer_integral: Float[Scalar, ""],
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_state_couplings: Float[Array, "num_modes num_states"],
):
    mode_couplings = mode_state_couplings[:, state_index]

    # calculate the state's diagonal values
    all_diagonal_values = jnp.repeat(
        transfer_integral, jnp.prod(jnp.array(mode_basis_sets))
    )

    # calculate the state's offdiagnoal values, arranged in a tuple for each mode
    all_mode_offdiagonal_values = calculate_state_offdiagonals(
        state_locality=False,
        mode_basis_sets=mode_basis_sets,
        mode_localities=mode_localities,
        mode_frequencies=mode_frequencies,
        mode_couplings=mode_couplings,
    )

    # build a state block with the diagonal values and tuple of offdiagonal values
    return build_state_block(
        all_diagonal_values=all_diagonal_values,
        all_mode_offdiagonal_values=all_mode_offdiagonal_values,
        mode_basis_sets=mode_basis_sets,
    )


def build_state_block(
    all_diagonal_values: Float[Array, "block_size"],
    all_mode_offdiagonal_values: tuple[Float[Array, "_"]],
    mode_basis_sets: Float[Array, "num_modes"],
) -> Float[Array, "block_size block_size"]:
    """Builds a single block of the full Hamiltonian matrix for a single state.

    Blocks have the form:
        - diagonal values go across the main diagonal
        - state blocks are broken into inner blocks for each state.
            From there, the diagonals one above and below the main diagonal (designated 'offdiagonals')
            are filled with values corresponding to the mode's offdiagonal components, where the diagonal's
            lower index is used in generation. Offdiagonal values are repeated to evenly fill the offdiagonal.

            For instance, two modes of basis set 3 will create the following block:

            [ d_0, m_0, 0,   n_0, 0,   0,   0,   0,   0
              m_0, d_1, m_1, 0,   n_0, 0,   0,   0,   0
              0,   m_1, d_2, 0,   0,   n_0, 0,   0,   0
              n_0, 0,   0,   d_3, m_0, 0,   n_1, 0,   0
              0,   n_0, 0,   m_0, d_4, m_1, 0,   n_1, 0
              0,   0,   n_0, 0,   m_1, d_5, 0,   0,   n_1
              0,   0,   0,   n_1, 0,   0, d_6,   m_0, 0
              0,   0,   0,   0,   n_1, 0,   m_0, d_7, m_1
              0,   0,   0,   0,   0,   n_1, 0,   m_1, d_8 ]

              Where d_i is the diagonal value at index i,
              m_i is the offdiagonal value for the last mode at subindex i
              n_i is the offdiagonal value for the last mode at subindex i
                (see calculate_mode_component functions)

    Args:
        all_diagonal_values (Float[Array, "block_size"]): list of all state diagonal values at each index.
        all_mode_offdiagonal_values (tuple[Float[Array, "_"]]): tuple containing an array of all offdiagonals at each subindex for each mode.
        mode_basis_sets (Float[Array, "num_modes"]): all basis sets for the state -- one for each mode.

    Returns:
        Float[Array, "block_size block_size"]: a constructed matrix block.
    """
    # start with an empty block of size 1
    block = jnp.zeros((1, 1))

    # run recursively for each mode, expanding by the basis set size and filling offdiagonal values
    for mode_basis_set, mode_offdiagonal_values in zip(
        reversed(mode_basis_sets), reversed(all_mode_offdiagonal_values)
    ):
        mode_offdiagonal_values = jnp.array(mode_offdiagonal_values)
        previous_block_size = len(block)

        # repeat each value to match the previous block size
        mode_offdiagonal_values = jnp.repeat(
            mode_offdiagonal_values, repeats=previous_block_size
        )

        # create a new block by repeating the previous block to match current basis set size
        new_block = jax.scipy.linalg.block_diag(*[block for _ in range(mode_basis_set)])

        # redefines the block by combining the new block with the new offdiagonal values
        block = (
            new_block
            + jnp.diag(mode_offdiagonal_values, k=previous_block_size)
            + jnp.diag(mode_offdiagonal_values, k=-previous_block_size)
        )

    # finally, fills the main diagonal of the full block
    block = block + jnp.diag(all_diagonal_values)
    return block


def calculate_state_local_diagonals(
    state_energy: Float[Scalar, ""],
    mode_frequencies: Float[Array, "num_modes"],
    mode_couplings: Float[Array, "num_modes"],
    mode_basis_sets: Int[Array, "num_modes"],
) -> Float[Array, "block_size"]:
    all_diagonal_values = [
        state_energy
        + jnp.sum(
            jnp.array(
                [
                    calculate_mode_local_diagonal_component(
                        component_index=component_index,
                        mode_frequency=mode_frequency,
                        mode_coupling=mode_coupling,
                    )
                    for component_index, mode_frequency, mode_coupling in zip(
                        mode_component_indices, mode_frequencies, mode_couplings
                    )
                ]
            )
        )
        for mode_component_indices in itertools.product(
            *[range(mode_basis_set) for mode_basis_set in mode_basis_sets]
        )
    ]

    return jnp.array(all_diagonal_values)


def calculate_state_offdiagonals(
    state_locality: Bool[Scalar, ""],
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_couplings: Float[Array, "num_modes"],
) -> tuple[Float[Array, "_"]]:
    all_mode_offdiagonal_values = tuple(
        (
            [
                calculate_mode_offdiagonal_component(
                    component_index=component_index,
                    mode_frequency=mode_frequency,
                    mode_coupling=mode_coupling,
                )
                for component_index in range(mode_basis_set - 1)
            ]
            if mode_locality == state_locality
            else jnp.zeros((mode_basis_set - 1,))
        )
        for mode_locality, mode_basis_set, mode_frequency, mode_coupling in zip(
            mode_localities, mode_basis_sets, mode_frequencies, mode_couplings
        )
    )

    return all_mode_offdiagonal_values


def calculate_mode_local_diagonal_component(
    component_index: Int[Scalar, ""],
    mode_frequency: Float[Scalar, ""],
    mode_coupling: Float[Scalar, ""],
) -> Float[Scalar, ""]:
    return mode_frequency * ((component_index + (1 / 2)) + (mode_coupling**2) / 2)


def calculate_mode_offdiagonal_component(
    component_index: Int[Scalar, ""],
    mode_frequency: Float[Array, ""],
    mode_coupling: Float[Array, ""],
) -> Float[Scalar, ""]:
    return mode_frequency * mode_coupling * jnp.sqrt((component_index + 1) / 2)
