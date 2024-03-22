import jax_dataclasses as jdc
from jaxtyping import Float, Int, Scalar

from abc import ABC, abstractmethod

from quantumspectra_2024.modules.absorption.AbsorptionSpectrum import AbsorptionSpectrum


@jdc.pytree_dataclass
class AbsorptionModel(ABC):
    """Represents a model for generating absorption spectra"""

    start_energy: jdc.Static[Float[Scalar, ""]] = 0.0
    end_energy: jdc.Static[Float[Scalar, ""]] = 20_000.0
    num_points: jdc.Static[Int[Scalar, ""]] = 2_001

    @abstractmethod
    def get_absorption(self) -> AbsorptionSpectrum:
        """Compute the absorption spectrum for the model."""
        pass
