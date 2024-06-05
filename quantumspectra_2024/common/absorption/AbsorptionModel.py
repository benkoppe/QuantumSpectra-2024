import jax_dataclasses as jdc

from abc import ABC, abstractmethod
import inspect

from quantumspectra_2024.common.absorption.AbsorptionSpectrum import (
    AbsorptionSpectrum,
)


@jdc.pytree_dataclass(kw_only=True)
class AbsorptionModel(ABC):
    """Represents a model for generating absorption spectra.

    All models include a `get_absorption` method that returns an `AbsorptionSpectrum` object.
    An `apply_electric_field` method is also included to apply an electric field to base models for Stark effect.

    Parameters
    ----------
    start_energy : float
        absorption spectrum's starting energy (wavenumbers).
    end_energy : float
        absorption spectrum's ending energy (wavenumbers).
    num_points : int
        absorption spectrum's number of points (unitless).
    """

    #: absorption spectrum's starting energy (wavenumbers).
    start_energy: float = 0.0
    #: absorption spectrum's ending energy (wavenumbers).
    end_energy: float = 20_000.0
    #: absorption spectrum's number of points (unitless).
    num_points: int = 2_001

    @classmethod
    def get_arguments(cls) -> list[str]:
        """Return a list of all argument names for the class.

        This is intended to ease object initialization of the class with keyword arguments.
        It can help with filtering a dictionary of parameters for initialization.

        Returns:
            list[str]: A list of all argument names for the class.
        """
        return list(inspect.signature(cls).parameters.keys())

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
        field_strength: float,
        field_delta_dipole: float,
        field_delta_polarizability: float,
    ) -> "AbsorptionModel":
        """Applies an electric field to the model. Returns a new instance of the model.

        Parameters
        ----------
        field_strength : float
            the strength of the electric field.
        field_delta_dipole : float
            the change in dipole moment due to the electric field.
        field_delta_polarizability : float
            the change in polarizability due to the electric field.

        Returns
        -------
        AbsorptionModel
            the model with the electric field applied.
        """
        raise NotImplementedError

    def __hash__(self):
        values = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, list):
                value = tuple(value)
            values.append(value)
        return hash(tuple(values))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        for field in self.__dataclass_fields__:
            value_self = getattr(self, field)
            value_other = getattr(other, field)
            if isinstance(value_self, list) and isinstance(value_other, list):
                if tuple(value_self) != tuple(value_other):
                    return False
            elif value_self != value_other:
                return False
        return True
