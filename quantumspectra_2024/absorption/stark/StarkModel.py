import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Scalar

from quantumspectra_2024.modules.absorption import (
    AbsorptionModel as Model,
    AbsorptionSpectrum,
)


@jdc.pytree_dataclass(kw_only=True)
class StarkModel(Model):
    """A general model for Stark absorption spectrum.

    Parameters
    ----------
    start_energy : Float[Scalar, ""]
        absorption spectrum's starting energy.
    end_energy : Float[Scalar, ""]
        absorption spectrum's ending energy.
    num_points : Int[Scalar, ""]
        absorption spectrum's number of points.

    neutral_submodel : Model
        parameterized neutral submodel to use in Stark effect calculation.

    positive_field_strength : Float[Scalar, ""]
        positive strength of the electric field.
    positive_field_sum_percent : Float[Scalar, ""]
        fraction of positive field strength to use in spectrum.
        Negative field strength is 1 - this value.

    field_delta_dipole : Float[Scalar, ""]
        change in dipole moment due to electric field.
    field_delta_polarizability : Float[Scalar, ""]
        change in polarizability due to electric field.
    """

    neutral_submodel: Model

    positive_field_strength: Float[Scalar, ""]
    positive_field_sum_percent: Float[Scalar, ""] = 0.5

    field_delta_dipole: Float[Scalar, ""]
    field_delta_polarizability: Float[Scalar, ""]

    def get_absorption(self) -> AbsorptionSpectrum:
        """Compute the absorption spectrum for the model.

        First computes absorption spectrum for the neutral submodel, and then for the charged submodels.
        Two charged submodels are computed, one with positive field strength and one with negative field strength.

        A half-sum is computed between the two charged submodels, with each submodel's intensities scaled by
        `positive_field_sum_percent` and `1 - positive_field_sum_percent`, respectively.

        Then, the neutral intensities are subtracted from the charged half-sum to get the electroabsorption spectrum.

        Returns
        -------
        AbsorptionSpectrum
            the model's parameterized absorption spectrum.
        """
        # get absorption spectrum sample energies (x values)
        neutral_submodel = self.get_neutral_submodel()
        positive_submodel = self.get_charged_submodel(field_strength_scalar=1.0)
        negative_submodel = self.get_charged_submodel(field_strength_scalar=-1.0)

        neutral_absorption = neutral_submodel.get_absorption()

        neutral_spectrum = neutral_absorption.intensities
        positive_spectrum = (
            positive_submodel.get_absorption().intensities
            * self.positive_field_sum_percent
        )
        negative_spectrum = negative_submodel.get_absorption().intensities * (
            1 - self.positive_field_sum_percent
        )

        charged_half_sum = jnp.sum(
            jnp.array([positive_spectrum, negative_spectrum]), axis=0
        )

        electroabsorption_spectrum = charged_half_sum - jnp.array(neutral_spectrum)

        return AbsorptionSpectrum(
            energies=neutral_absorption.energies,
            intensities=electroabsorption_spectrum,
        )

    def get_neutral_submodel(self) -> Model:
        """Returns the netural submodel with the Stark effect applied.

        This method replaces the neutral submodel's point values with the Stark model's point values.
        No other changes are made to the neutral submodel.

        Returns
        -------
        Model
            the neutral submodel for Stark calculations.
        """
        # replace neutral submodel with own point values
        neutral_submodel = jdc.replace(
            self.neutral_submodel,
            start_energy=self.start_energy,
            end_energy=self.end_energy,
            num_points=self.num_points,
        )

        return neutral_submodel

    def get_charged_submodel(self, field_strength_scalar: Float[Scalar, ""]) -> Model:
        """Returns a charged submodel with the Stark effect applied.

        This method starts with the neutral submodel from `get_neutral_submodel` and applies the Stark effect.
        The Stark effect is applied through the Model's `apply_electric_field` method.
        Field strength is multiplied by `field_strength_scalar`, and all other values are inputted into the method exactly.

        Parameters
        ----------
        field_strength_scalar : Float[Scalar, ""]
            scalar to multiply the field strength by.

        Returns
        -------
        Model
            the charged submodel for Stark calculations.
        """
        # get neutral submodel
        neutral_submodel = self.get_neutral_submodel()

        # apply specified charge to neutral submodel
        field_strength = float(self.positive_field_strength * field_strength_scalar)
        field_delta_dipole = float(self.field_delta_dipole)
        field_delta_polarizability = float(self.field_delta_polarizability)

        charged_submodel = neutral_submodel.apply_electric_field(
            field_strength=field_strength,
            field_delta_dipole=field_delta_dipole,
            field_delta_polarizability=field_delta_polarizability,
        )

        return charged_submodel

    def apply_electric_field(*_) -> None:
        """Applies an electric field to the model. Stark effect is not configured to apply external electric fields.

        Raises
        ------
        NotImplementedError
            StarkModel does not support apply_electric_field.
        """
        raise NotImplementedError("StarkModel does not support apply_electric_field")
