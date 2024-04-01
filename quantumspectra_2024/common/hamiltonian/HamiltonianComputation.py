import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Bool, Array


def diagonalize_matrix(matrix: Float[Array, "matrix_size matrix_size"]) -> tuple[
    Float[Array, "matrix_size"],
    Float[Array, "matrix_size matrix_size"],
]:
    """Diagonalizes a matrix and returns the eigenvalues and eigenvectors in a tuple.

    Parameters
    ----------
    matrix : Float[Array, "matrix_size matrix_size"]
        a matrix to diagonalize.

    Returns
    -------
    tuple[Float[Array, "matrix_size"], Float[Array, "matrix_size matrix_size"]]
        tuple containing the eigenvalues and eigenvectors.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def build_matrix(
    state_energies: Float[Array, "num_states"],
    transfer_integral: float,
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_state_couplings: Float[Array, "num_modes num_states"],
) -> Float[Array, "matrix_size matrix_size"]:
    """Builds a full Hamiltonian matrix for a system of states and modes.

    Hamiltonians are built with a combination of blocks. Each block represents a single state.
    States along the diagonal are so-called 'local' states, while offdiagonal states are 'nonlocal'.
    See documentation for `build_local_state_block` and `build_nonlocal_state_block` for more information.

    Notes
    -----
    TODO:
    * Implement multiple transfer integrals for more than two states.

    Parameters
    ----------
    state_energies : Float[Array, "num_states"]
        energies of all local states.
    transfer_integral : float
        transfer integral between states.
    mode_basis_sets : Int[Array, "num_modes"]
        basis set size per mode.
    mode_localities : Bool[Array, "num_modes"]
        locality of each mode.
    mode_frequencies : Float[Array, "num_modes"]
        frequency per mode.
    mode_state_couplings : Float[Array, "num_modes num_states"]
        coupling per mode and local state.

    Returns
    -------
    Float[Array, "matrix_size matrix_size"]
        fully constructed matrix in `jax` array.
    """
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
    state_index: int,
    state_energies: Float[Array, "num_states"],
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_state_couplings: Float[Array, "num_modes num_states"],
) -> Float[Array, "block_size block_size"]:
    """Builds a local state block.

    To see how block are built, see `build_state_block`.
    To build a local state block, the following steps are taken.
    * Calculate the state's diagonal values with `calculate_state_local_diagonals`.
    * Calculate the state's offdiagonal values with `calculate_state_offdiagonals`.
    * Build the state block.

    Parameters
    ----------
    state_index : int
        index of the state.
    state_energies : Float[Array, "num_states"]
        energies of all local states.
    mode_basis_sets : Int[Array, "num_modes"]
        basis set size per mode.
    mode_localities : Bool[Array, "num_modes"]
        locality of each mode.
    mode_frequencies : Float[Array, "num_modes"]
        frequency per mode.
    mode_state_couplings : Float[Array, "num_modes num_states"]
        coupling per mode and local state.

    Returns
    -------
    Float[Array, "block_size block_size"]
        constructed matrix block.
    """
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
    state_index: int,
    transfer_integral: float,
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_state_couplings: Float[Array, "num_modes num_states"],
):
    """Builds a nonlocal state block.

    To see how blocks are built, see `build_state_block`.
    To build a nonlocal state block, the following steps are taken.
    * The state's offdiagonal values are all set to the transfer integral.
    * Calculate the state's diagonal values with `calculate_state_local_diagonals`.
    * Build the state block.

    Parameters
    ----------
    state_index : int
        index of the state.
    transfer_integral : float
        transfer integral between states.
    mode_basis_sets : Int[Array, "num_modes"]
        basis set size per mode.
    mode_localities : Bool[Array, "num_modes"]
        locality of each mode.
    mode_frequencies : Float[Array, "num_modes"]
        frequency per mode.
    mode_state_couplings : Float[Array, "num_modes num_states"]
        coupling per mode and local state.

    Returns
    -------
    Float[Array, "block_size block_size"]
        constructed matrix block.
    """
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

    Parameters
    ----------
    all_diagonal_values : Float[Array, "block_size"]
        all state diagonal values.
    all_mode_offdiagonal_values : tuple[Float[Array, "_"]]
        all offdiagonal values for the state block per mode.
    mode_basis_sets : Float[Array, "num_modes"]
        basis set sizes per mode.

    Returns
    -------
    Float[Array, "block_size block_size"]
        constructed matrix block.
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
    state_energy: float,
    mode_frequencies: Float[Array, "num_modes"],
    mode_couplings: Float[Array, "num_modes"],
    mode_basis_sets: Int[Array, "num_modes"],
) -> Float[Array, "block_size"]:
    """Calculate all diagonal values for a state block in a single array. Only calculated for local blocks.

    Diagonal values are calculated with a sum of contributions from each mode, plus the state energy.
    Contributions for each mode are summed with all combinations of values between 1 and their basis set size - 1.
    To see how individual mode contributions are calculated, see `calculate_mode_local_diagonal_component`.

    Parameters
    ----------
    state_energy : float
        energy of the state.
    mode_frequencies : Float[Array, "num_modes"]
        frequencies of each mode.
    mode_couplings : Float[Array, "num_modes"]
        couplings of each mode.
    mode_basis_sets : Int[Array, "num_modes"]
        basis set size of each mode.

    Returns
    -------
    Float[Array, "block_size"]
        all diagonal values for the state block.
    """
    parallelized_contribution_func = jax.vmap(
        calculate_mode_local_diagonal_component, in_axes=(0, None, None)
    )

    mode_diagonal_contributions = [
        parallelized_contribution_func(
            jnp.arange(mode_basis_set),
            mode_frequency,
            mode_coupling,
        )
        for mode_basis_set, mode_frequency, mode_coupling in zip(
            mode_basis_sets, mode_frequencies, mode_couplings
        )
    ]

    sum_contribution_combinations = outer_sum(*mode_diagonal_contributions).flatten()

    all_diagonal_values = state_energy + sum_contribution_combinations

    return all_diagonal_values


def calculate_state_offdiagonals(
    state_locality: bool,
    mode_basis_sets: Int[Array, "num_modes"],
    mode_localities: Bool[Array, "num_modes"],
    mode_frequencies: Float[Array, "num_modes"],
    mode_couplings: Float[Array, "num_modes"],
) -> tuple[Float[Array, "_mode_offdiagonal_size"]]:
    """Calculate all unqiue offdiagonal values for a state block per mode. Each mode's set of offdiagonals is contained in a tuple.

    Offdiagonal values are an array of calculated values that range from 1 to the basis set size of the mode - 1.
    Thus, each array in the tuple will have a length of `mode_basis_set`.
    If the mode doesn't match the locality of the state, the offdiagonals are all set to zero.
    To see how individual mode offdiagonal values are calculated, see `calculate_mode_offdiagonal_component`.

    Parameters
    ----------
    state_locality : bool
        the locality of the state.
    mode_basis_sets : Int[Array, "num_modes"]
        basis set size of each mode.
    mode_localities : Bool[Array, "num_modes"]
        locality of each mode.
    mode_frequencies : Float[Array, "num_modes"]
        frequencies of each mode.
    mode_couplings : Float[Array, "num_modes"]
        couplings of each mode.

    Returns
    -------
    tuple[Float[Array, "_mode_offdiagonal_size"]]
        all offdiagonal values for the state block per mode.
    """
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
    component_index: int,
    mode_frequency: float,
    mode_coupling: float,
) -> float:
    """Computes a single diagonal contribution component for a mode.

    Diagonal contributions are calculated as follows:
        * The mode frequency is multiplied by the component index plus one half
        * This value is then multiplied by the square of the mode coupling, divided by two.

    Parameters
    ----------
    component_index : int
        the index of the component.
    mode_frequency : float
        the frequency of the mode.
    mode_coupling : float
        the coupling of the mode.

    Returns
    -------
    float
        the computed diagonal contribution component.
    """
    return mode_frequency * ((component_index + (1 / 2)) + (mode_coupling**2) / 2)


def calculate_mode_offdiagonal_component(
    component_index: int,
    mode_frequency: float,
    mode_coupling: float,
) -> float:
    """Computes a single offdiagonal contribution component for a mode.

    Offdiagonal contributions are calculated as follows:
        * The mode frequency is multiplied by the square root of the component index plus one, divided by two.
        * This value is then multiplied by the mode coupling.

    Parameters
    ----------
    component_index : int
        the index of the component.
    mode_frequency : float
        the frequency of the mode.
    mode_coupling : float
        the coupling of the mode.

    Returns
    -------
    float
        the computed offdiagonal contribution component.
    """
    return mode_frequency * mode_coupling * jnp.sqrt((component_index + 1) / 2)


def outer_sum(*arrays: tuple[Float[Array, "*"]]) -> Float[Array, "*"]:
    """
    Computes the outer sum of multiple JAX arrays.

    This function takes multiple JAX arrays as input and computes their outer sum. It starts with the first array and
    iteratively adds each subsequent array to it in a way that's similar to computing the outer product, but with
    summation instead. This is done by reshaping the arrays for broadcasting, ensuring dimensions are aligned correctly
    for the sum.

    Parameters
    ----------
    *arrays: tuple[Float[Array, "*"]]
        Variable number of JAX array arguments.
        Each array should be compatible for broadcasting.
        There should be at least one array passed to this function.

    Returns
    -------
    Float[Array, "*"]
        A JAX array containing the outer sum of the input arrays.

    Raises
    ------
    ValueError
        If no arrays are provided as input.

    Examples
    --------
        >>> import jax.numpy as jnp
        >>> a = jnp.array([1, 2])
        >>> b = jnp.array([3, 4])
        >>> outer_sum(a, b)
        DeviceArray([[4, 5],
                     [5, 6]], dtype=int32)

    Notes
    -----
        The function requires at least one input array and all input arrays must be compatible for broadcasting
        following the JAX rules.
    """
    # Ensure there is at least one array
    if not arrays:
        raise ValueError("At least one array is required")

    # Start with the first array, reshaping it to have a new axis for each additional array
    result = arrays[0].reshape(arrays[0].shape + (1,) * (len(arrays) - 1))

    # Iteratively add each subsequent array, reshaping appropriately for broadcasting
    for i, arr in enumerate(arrays[1:], 1):
        # The new shape has 1's in all positions except the current dimension being added
        new_shape = (1,) * i + arr.shape + (1,) * (len(arrays) - i - 1)
        result += arr.reshape(new_shape)

    return result
