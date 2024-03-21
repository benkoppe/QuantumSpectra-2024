import jax


class AbsorptionSpectrum:
    """Represents an absorption spectrum. Outputted by all `AbsorptionModel` subclasses."""

    energies: jax.Array
    intensities: jax.Array

    def save_file(self, filename: str) -> None:
        """Save the absorption spectrum to a file.

        Args:
            filename (str): output filename
        """
        pass
