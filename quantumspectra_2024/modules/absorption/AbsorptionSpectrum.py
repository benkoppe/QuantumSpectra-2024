import numpy as np
import matplotlib.pyplot as plt

import jax_dataclasses as jdc
from jaxtyping import Float, Array


@jdc.pytree_dataclass(kw_only=True)
class AbsorptionSpectrum:
    """Represents an absorption spectrum. Outputted by all `AbsorptionModel` subclasses.

    Args:
        energies (Float[Array, "num_points"]): the x values of the absorption spectrum.
        intensities (Float[Array, "num_points"]): the y values of the absorption spectrum.
    """

    energies: Float[Array, "num_points"]
    intensities: Float[Array, "num_points"]

    def save_data(self, filename: str) -> None:
        """Save the absorption spectrum data to a file.

        Args:
            filename (str): output filename.
        """
        combined_data = np.column_stack(
            (np.array(self.energies), np.array(self.intensities))
        )

        np.savetxt(filename, combined_data, delimiter=",")

    def save_plot(self, filename: str) -> None:
        """Save the absorption spectrum plot to a file.

        Args:
            filename (str): output filename.
        """
        plt.plot(self.energies, self.intensities)
        plt.xlabel("Energy (cm^-1)")
        plt.ylabel("Intensity")
        plt.savefig(filename)
        plt.close()
