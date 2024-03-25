import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, Scalar


def broaden_peaks(
    sample_points: Float[Array, "num_points"],
    peak_energies: Float[Array, "num_peaks"],
    peak_intensities: Float[Array, "num_peaks"],
    distribution_broadening: Float[Scalar, ""],
) -> Float[Array, "num_points"]:
    """Broadens a set of peaks with given sample points into an absorption spectrum.

    This function first computes a set of Gaussian distributions centered at each peak energy, with a specified broadening.
    Then, each spectrum is scaled by the peak intensity and summed to form the final absorption spectrum.

    See `compute_gaussians` for more information on the Gaussian distributions.

    Args:
        sample_points (Float[Array, "num_points"]): Sample points at which the absorption spectrum will be computed.
        peak_energies (Float[Array, "num_peaks"]): Peak energies corresponding to the center of each Gaussian distribution.
        peak_intensities (Float[Array, "num_peaks"]): Peak intensities to scale each Gaussian distribution.
        distribution_broadening (Float[Scalar, ""]): Standard deviation of the Gaussian distributions.

    Returns:
        Float[Array, "num_points"]: Computed absorption spectrum.
    """
    gaussians = compute_gaussians(
        sample_points=sample_points,
        peak_energies=peak_energies,
        distribution_broadening=distribution_broadening,
    )
    return jnp.sum(peak_intensities * gaussians, axis=1)


@jax.jit
def compute_gaussians(
    sample_points: Float[Array, "num_points"],
    peak_energies: Float[Array, "num_peaks"],
    distribution_broadening: Float[Scalar, ""],
) -> Float[Array, "num_points num_peaks"]:
    """Computes the Gaussian distributions for a set of sample points and peak energies.

    This function calculates the value of Gaussian functions for each combination of
    sample points and peak energies, considering a specified distribution broadening.
    It is `jax` JIT-compiled to improve performance.

    Args:
        sample_points (Float[Array, "num_points"]): Sample points at which the Gaussian functions
            will be evaluated. Each point represents a position on the x-axis of the Gaussian distribution.
        peak_energies (Float[Array, "num_peaks"]): Peak energies corresponding
            to the mean (center) of each Gaussian distribution. Each peak energy defines
            a different Gaussian function to be evaluated at the sample points.
        distribution_broadening (Float[Scalar, ""]): A scalar value representing the
            standard deviation (sigma) of the Gaussian distributions, which determines
            the broadening of the distributions. This value is the same for all
            Gaussian distributions computed by this function.

    Returns:
        Float[Array, "num_points num_peaks"]: Computed Gaussian values for each combination of sample points and peak energies.
    """
    return (1.0 / (jnp.sqrt(jnp.pi) * distribution_broadening)) * jnp.exp(
        -jnp.power(
            (sample_points[:, jnp.newaxis] - peak_energies) / distribution_broadening, 2
        )
    )


def compute_peaks(
    eigenvalues: Float[Array, "matrix_size"],
    eigenvectors: Float[Array, "matrix_size matrix_size"],
    transfer_integral: Float[Scalar, ""],
    temperature_kelvin: Float[Scalar, ""],
) -> tuple[Float[Array, "num_peaks"], Float[Array, "num_peaks"]]:
    """Computes the absorption spectrum peak energies and intensities for a two-state system.

    Diagonalized eigenvalues and eigenvectors are first used to compute peak energies and intensities.
    See `compute_peak_energies` and `compute_peak_intensities` for more information.

    Intensities are scaled by probability scalars if the temperature is not 0.
    See `compute_peak_probability_scalars` for more information.

    Energies and intensities are then constrained only to pair combinations between a range of first eigenvectors and all other elevated energy levels.
    First eigenvector range is 1 with 0 temperature, or 50 with non-zero temperature.
    See `filter_peaks` for more information.

    Returns:
        tuple[Float[Array, "num_peaks"], Float[Array, "num_peaks"]]: Computed absorption spectrum peak energies and intensities.
    """
    # compute all possible peak energies and intensities
    energies = compute_peak_energies(eigenvalues=eigenvalues)
    intensities = compute_peak_intensities(
        eigenvectors=eigenvectors, transfer_integral=transfer_integral
    )

    # compute temperature to wavenumbers
    temperature_wavenumbers = temperature_kelvin * 0.695028

    # define range of eigenvectors used in pair combinations
    first_eigenvector_range = jax.lax.cond(
        (temperature_wavenumbers == 0),
        lambda: 1,
        lambda: min(50, len(eigenvalues)),
    )

    # scale intensities by probability scalars if temperature is not 0
    scaled_intensities = jax.lax.cond(
        (temperature_wavenumbers == 0),
        lambda: intensities,
        lambda: intensities
        * compute_peak_probability_scalars(
            eigenvalues=eigenvalues, temperature_wavenumbers=temperature_wavenumbers
        )[:, None],
    )

    # filter computed intensities and energies to retrieve pair combinations between first_eigenvector_range and all other elevated energy levels
    filtered_energies, filtered_intensities = filter_peaks(
        peak_energies=energies,
        peak_intensities=scaled_intensities,
        first_eigenvector_range=first_eigenvector_range,
    )

    return filtered_energies, filtered_intensities


def compute_peak_energies(
    eigenvalues: Float[Array, "matrix_size"],
) -> Float[Array, "matrix_size/2 matrix_size"]:
    """Compute all raw spectrum peak energy values.

    Energy values are the differences between two state's eigenvalues.
    This function computes all differences between eigenvalues and returns a matrix of differences.
    The matrix follows the form where index i,j represents the difference between the ith and jth eigenvalues.

    Because only the first state will be used for pairs, only the first half of the differences are necessary and returned.

    Returns:
        Float[Array, "matrix_size/2 matrix_size"]: difference matrix of eigenvalues.
    """
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
    """Compute all raw spectrum peak intensity values.

    Intensity values are the dot product of two sets of sliced eigenvectors, squared.
    This function computes all squared dot products between pairs of eigenvectors and returns a matrix of intensities.
    The matrix follows the form where index i,j represents the intensity between the ith and jth eigenvectors.

    Because only the first state will be used for pairs, only the first half of the intensities are necessary and returned.

    The second eigenvalue in the computation is sliced in half depending on `transfer_integral` value:
        * if `transfer_integral` is not 0, the top half of the eigenvectors are used
        * if `transfer_integral` is 0, the bottom half of the eigenvectors are used

    Returns:
        Float[Array, "matrix_size/2 matrix_size"]: intensity matrix of eigenvectors.
    """
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
    """Computes a probability scalars for peak intensities.

    Computes a normalized exponential probability distribution based on the differences
    between the first half of the eigenvalues and the first eigenvalue, scaled by a given temperature.

    This function first calculates the differences between the first half of eigenvalues and the first eigenvalue.
    Then, it computes the exponential of the negative of these differences, divided by the temperature in wavenumbers.
    Finally, it normalizes these exponentials to sum to one, forming a probability distribution.

    The probability scalars are returned as a vector of the same size as the first half of the eigenvalues.
    The probability scalars are used to scale the intensities of the peaks in the absorption spectrum.

    Returns:
        Float[Array, "matrix_size/2"]: probability scalars for peak intensities.
    """
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
) -> tuple[Float[Array, "num_peaks"], Float[Array, "num_peaks"]]:
    """Filters peak energies and intensities by selecting unique and correct pair combinations and filtering negative values.

    This function uses upper-triangular matrix indices to select only peaks that have been computed with
    unique pair combinations between first eigenvectors in the given `first_eigenvector_range` and all other elevated energy levels.
    It also filters out negative energy or intensity values.

    Args:
        peak_energies (Float[Array, "matrix_size/2 matrix_size"]): peak energies computed by `compute_peak_energies`.
        peak_intensities (Float[Array, "matrix_size/2 matrix_size"]): peak intensities computed by `compute_peak_intensities`.
        first_eigenvector_range (Int[Scalar, ""]): the range of first eigenvectors to consider for pair combinations.

    Returns:
        tuple[Float[Array, "num_peaks"], Float[Array, "num_peaks"]]: filtered peak energies and intensities.
    """
    # get upper-triangular indices starting from the first off-diagonal
    triu_indices = jnp.triu_indices(first_eigenvector_range, k=1, m=len(peak_energies))

    # define the filtering mask
    mask = ((peak_intensities >= 0) | (peak_energies >= 0))[triu_indices]

    # filter energies and intensities
    filtered_peak_energies = peak_energies[triu_indices][mask]
    filtered_peak_intensities = peak_intensities[triu_indices][mask]

    # this filters the arrays such that only unique pair combinations are considered, and
    # such that the intensities and energies are non-negative

    return filtered_peak_energies, filtered_peak_intensities
