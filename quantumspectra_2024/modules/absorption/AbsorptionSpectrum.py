import jax_dataclasses as jdc
from jaxtyping import Float, Array


@jdc.pytree_dataclass(kw_only=True)
class AbsorptionSpectrum:
    """Represents an absorption spectrum. Outputted by all `AbsorptionModel` subclasses."""

    energies: Float[Array, "num_points"]
    intensities: Float[Array, "num_points"]

    def save_file(self, filename: str) -> None:
        """Save the absorption spectrum to a file.

        Args:
            filename (str): output filename
        """
        pass
