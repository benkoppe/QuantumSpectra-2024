import jax_dataclasses as jdc
from jaxtyping import Float, Int, Array, Scalar

from quantumspectra_2024.modules.absorption.AbsorptionModel import (
    AbsorptionModel as Model,
)


@jdc.pytree_dataclass
class TwoStateModel(Model):
    """A two-state quantum mechanical model for absorption spectra."""

    temperature_kelvin: Float[Scalar, ""]
    broadening: Float[Scalar, ""]

    basis_sets: jdc.Static[Int[Array, "2"]]

    transfer_integral: Float[Scalar, ""]
    energy_gap: Float[Scalar, ""]

    mode_frequencies: Float[Array, "num_modes"]
    mode_couplings: Float[Array, "num_modes"]
