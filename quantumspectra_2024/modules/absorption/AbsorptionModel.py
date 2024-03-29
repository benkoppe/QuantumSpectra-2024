import jax_dataclasses as jdc
from jaxtyping import Float, Int, Scalar

from abc import ABC, abstractmethod

from quantumspectra_2024.modules.absorption.AbsorptionSpectrum import AbsorptionSpectrum


@jdc.pytree_dataclass(kw_only=True)
class AbsorptionModel(ABC):
    """Represents a model for generating absorption spectra.

    All models include a `get_absorption` method that returns an `AbsorptionSpectrum` object.
    An `apply_electric_field` method is also included to apply an electric field to base models for Stark effect.

    Parameters
    ----------
    start_energy : Float[Scalar, ""]
        absorption spectrum's starting energy.
    end_energy : Float[Scalar, ""]
        absorption spectrum's ending energy.
    num_points : Int[Scalar, ""]
        absorption spectrum's number of points.
    """

    #: Float[Scalar, ""]: absorption spectrum's starting energy.
    start_energy: jdc.Static[Float[Scalar, ""]] = 0.0
    #: Float[Scalar, ""]: absorption spectrum's ending energy.
    end_energy: jdc.Static[Float[Scalar, ""]] = 20_000.0
    #: Int[Scalar, ""]: absorption spectrum's number of points.
    num_points: jdc.Static[Int[Scalar, ""]] = 2_001

    @abstractmethod
    def get_absorption(self) -> AbsorptionSpectrum:
        """Compute the absorption spectrum for the model.

        Returns
        -------
        AbsorptionSpectrum
            the model's parameterized absorption spectrum.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_electric_field(
        field_strength: Float[Scalar, ""],
        field_delta_dipole: Float[Scalar, ""],
        field_delta_polarizability: Float[Scalar, ""],
    ) -> "AbsorptionModel":
        """Applies an electric field to the model. Returns a new instance of the model.

        Parameters
        ----------
        field_strength : Float[Scalar, ""]
            the strength of the electric field.
        field_delta_dipole : Float[Scalar, ""]
            the change in dipole moment due to the electric field.
        field_delta_polarizability : Float[Scalar, ""]
            the change in polarizability due to the electric field.

        Returns
        -------
        AbsorptionModel
            the model with the electric field applied.
        """
        raise NotImplementedError
