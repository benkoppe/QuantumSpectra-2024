import numpy as np
import matplotlib.pyplot as plt

import jax_dataclasses as jdc
from jaxtyping import Float, Array


@jdc.pytree_dataclass(kw_only=True)
class AbsorptionSpectrum:
    """Represents an absorption spectrum. Outputted by all `AbsorptionModel` subclasses.

    Parameters
    ----------
    energies : Float[Array, "num_points"]
        the x values of the absorption spectrum.
    intensities : Float[Array, "num_points"]
        the y values of the absorption spectrum.
    """

    energies: Float[Array, "num_points"]
    intensities: Float[Array, "num_points"]

    def cut_bounds(
        self, start_energy: float = None, end_energy: float = None
    ) -> "AbsorptionSpectrum":
        """Cut the absorption spectrum to a specific energy range.

        Parameters
        ----------
        start_energy : float
            the starting energy of the cut.
        end_energy : float
            the ending energy of the cut.

        Returns
        -------
        AbsorptionSpectrum
            the cut absorption spectrum.
        """
        if start_energy is None:
            start_energy = self.energies[0]
        if end_energy is None:
            end_energy = self.energies[-1]

        mask = (self.energies >= start_energy) & (self.energies <= end_energy)

        return AbsorptionSpectrum(
            energies=self.energies[mask], intensities=self.intensities[mask]
        )

    def save_data(self, filename: str) -> None:
        """Save the absorption spectrum data to a file.

        Parameters
        ----------
        filename : str
            output filename.
        """
        combined_data = np.column_stack(
            (np.array(self.energies), np.array(self.intensities))
        )

        np.savetxt(filename, combined_data, delimiter=",")

    def save_plot(self, filename: str) -> None:
        """Save the absorption spectrum plot to a file.

        Parameters
        ----------
        filename : str
            output filename.
        """
        plt.plot(self.energies, self.intensities)
        plt.xlabel("Energy (cm^-1)")
        plt.ylabel("Intensity")
        plt.savefig(filename)
        plt.close()
