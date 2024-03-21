import jax_dataclasses as jdc

from abc import ABC


@jdc.pytree_dataclass
class AbsorptionModel(ABC):
    """Represents a model for generating absorption spectra"""

    start_energy: jdc.Static[float] = 0.0
    end_energy: jdc.Static[float] = 20_000.0
    num_points: jdc.Static[int] = 2_001
