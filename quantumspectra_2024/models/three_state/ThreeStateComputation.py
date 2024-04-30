import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array

from quantumspectra_2024.common.hamiltonian.HamiltonianComputation import (
    calculate_state_local_diagonals,
)


def compute_peaks(
    two_state_eigenvalues: Float[Array, "matrix_size"],
    two_state_eigenvectors: Float[Array, "matrix_size matrix_size"],
    d_LE: float,
    d_CT: float,
    temperature_kelvin: float,
) -> tuple[Float[Array, "num_peaks"], Float[Array, "num_peaks"]]:
    """Computes the absorption spectrum peak energies and intensities for a two-state system.

    Diagonalized eigenvalues and eigenvectors are first used to compute peak energies and intensities.
    See `compute_peak_energies` and `compute_peak_intensities` for more information.

    Intensities are scaled by probability scalars if the temperature is not 0.
    See `compute_peak_probability_scalars` for more information.

    Energies and intensities are then constrained only to pair combinations between a range of first eigenvectors and all other elevated energy levels.
    First eigenvector range is 1 with 0 temperature, or 50 with non-zero temperature.
    See `filter_peaks` for more information.

    Parameters
    ----------
    eigenvalues : Float[Array, "matrix_size"]
        Eigenvalues of the Hamiltonian.
    eigenvectors : Float[Array, "matrix_size matrix_size"]
        Eigenvectors of the Hamiltonian.
    transfer_integral : float
        Transfer integral between the two states.
    temperature_kelvin : float
        System's temperature in Kelvin.

    Returns
    -------
    tuple[Float[Array, "num_peaks"], Float[Array, "num_peaks"]]
        Computed absorption spectrum peak energies and intensities.
    """
    # compute all possible peak energies and intensities
    energies = two_state_eigenvalues
    intensities_matrix = compute_peak_pure_intensities(
        two_state_eigenvectors=two_state_eigenvectors,
        d_LE=d_LE,
        d_CT=d_CT,
    )

    # compute temperature to wavenumbers
    temperature_wavenumbers = temperature_kelvin * 0.695028

    # scale intensities by probability scalars
    probability_scalars = compute_peak_probability_scalars(
        two_state_eigenvalues=two_state_eigenvalues,
        temperature_wavenumbers=temperature_wavenumbers,
    )
    intensities = pure_intensities * probability_scalars

    return energies, intensities


def compute_peak_pure_intensities(
    two_state_eigenvectors: Float[Array, "matrix_size matrix_size"],
    d_LE: float,
    d_CT: float,
) -> Float[Array, "matrix_size/2 matrix_size/2"]:
    """Compute all raw spectrum peak intensity values.

    Intensity values are the dot product of two sets of sliced eigenvectors, squared.
    This function computes all squared dot products between pairs of eigenvectors and returns a matrix of intensities.
    The matrix follows the form where index i,j represents the intensity between the ith and jth eigenvectors.

    Because only the first state will be used for pairs, only the first half of the intensities are necessary and returned.

    The second eigenvalue in the computation is sliced in half depending on `transfer_integral` value:
        * if `transfer_integral` is not 0, the top half of the eigenvectors are used
        * if `transfer_integral` is 0, the bottom half of the eigenvectors are used

    Parameters
    ----------
    eigenvectors : Float[Array, "matrix_size matrix_size"]
        Eigenvectors of the Hamiltonian.
    transfer_integral : float
        Transfer integral between the two states.

    Returns
    -------
    Float[Array, "matrix_size/2"]
        intensity matrix of eigenvectors.
    """
    # half the size of a dimension of the eignvectors
    half_size = len(two_state_eigenvectors) // 2

    # LE eigenvectors are the upper half of the eigenvectors, size prod(mode_basis_sets)
    LE_eigenvectors = two_state_eigenvectors[:, :half_size]

    # CT eigenvectors are the lower half of the eigenvectors, size prod(mode_basis_sets)
    CT_eigenvectors = two_state_eigenvectors[:, half_size:]

    # cut both eigenvectors in half as well, as only the first half will be used
    LE_eigenvectors = LE_eigenvectors[:half_size]
    CT_eigenvectors = CT_eigenvectors[:half_size]

    # compute pure intensity matrix for all pairs of eigenvectors
    pure_intensities = jnp.abs(d_LE * LE_eigenvectors + d_CT * CT_eigenvectors) ** 2

    # return pure intensities (not scaled by probabilities)
    # computes a matrix where slot i, j is intensity between ith eigenvector and jth ground state
    return pure_intensities.T  # transpose necessary to match above description


def compute_peak_energies(
    two_state_eigenvalues: Float[Array, "matrix_size"],
    ground_state_energy: float,
    mode_frequencies: Float[Array, "num_modes"],
    mode_basis_sets: Int[Array, "num_modes"],
):
    ground_state_energies = calculate_state_local_diagonals(
        state_energy=ground_state_energy,
        mode_frequencies=mode_frequencies,
        mode_couplings=jnp.zeros_like(mode_frequencies),
        mode_basis_sets=mode_basis_sets,
    )

    two_state_eigenvalues_col = two_state_eigenvalues[:, jnp.newaxis]
    ground_state_energies_row = ground_state_energies[jnp.newaxis, :]

    # compute a difference matrix where slot i, j is difference between ith eigenvalue and jth ground state energy
    differences_matrix = two_state_eigenvalues_col - ground_state_energies_row
    return differences_matrix


def compute_peak_probability_scalars(
    two_state_eigenvalues: Float[Array, "matrix_size"],
    temperature_wavenumbers: float,
):
    # take exponentials of the negative eigenvalues divided by the temperature
    exponentials = jnp.exp(-two_state_eigenvalues / temperature_wavenumbers)

    # return normalized exponential of scaled differences
    return exponentials / jnp.sum(exponentials)
