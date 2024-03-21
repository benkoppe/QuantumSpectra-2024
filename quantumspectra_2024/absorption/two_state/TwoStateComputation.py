import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, Scalar

from quantumspectra_2024.modules.hamiltonian import HamiltonianModel


def compute_peak_energies(
    eigenvalues: Float[Array, "matrix_size"],
) -> Float[Array, "matrix_size/2 matrix_size"]:
    # half the number of eigenvalues
    half_size = len(eigenvalues) // 2

    # reshape eigenvalues to column vector
    eigenvalues_col = eigenvalues[:, jnp.newaxis]
    # subtract the column vector from the transpose to get the difference matrix
    differences_matrix = eigenvalues - eigenvalues_col

    # only the first half is necessary
    return differences_matrix[:half_size]


def compute_peak_intensities(
    eigenvectors: Float[Array, "matrix_size matrix_size"],
    transfer_integral: Float[Scalar, ""],
) -> Float[Array, "matrix_size/2 matrix_size"]:
    # half the size of a dimension of the eignvectors
    half_size = len(eigenvectors) // 2

    # slice the first dimension of the eigenvectors -> an array of half-sized vectors from the bottom of the eigenvectors
    vector_slices_1 = eigenvectors[:half_size, :]
    # slicing for the second set depends on t value -> if t is 0, slice the top half otherwise slice like vector_slices_1
    vector_slices_2 = jax.lax.cond(
        (transfer_integral == 0),
        lambda: eigenvectors[half_size:, :],
        lambda: vector_slices_1,
    )

    # compute dot product of the two sets of eigenvectors, and square result
    # first set is transposed to make result match dot products between all combinations of vectors
    intensities_matrix = jnp.dot(vector_slices_1.T, vector_slices_2) ** 2

    # only the first half is necessary
    return intensities_matrix[:half_size]


def compute_peak_probability_scalars(
    eigenvalues: Float[Array, "matrix_size"],
    temperature_wavenumbers: Float[Scalar, ""],
) -> Float[Array, "matrix_size/2"]:
    # half the number of eigenvalues
    half_size = len(eigenvalues) // 2

    # find differences between the first half of the eigenvalues and the first eigenvalue
    differences = eigenvalues[:half_size] - eigenvalues[0]
    # take exponentials of the negative differences divided by the temperature
    exponentials = jnp.exp(-differences / temperature_wavenumbers)

    # return normalized exponential of scaled differences
    return exponentials / jnp.sum(exponentials)


def filter_peaks(
    peak_energies: Float[Array, "matrix_size/2 matrix_size"],
    peak_intensities: Float[Array, "matrix_size/2 matrix_size"],
    first_eigenvector_range: Int[Scalar, ""],
):
    pass
