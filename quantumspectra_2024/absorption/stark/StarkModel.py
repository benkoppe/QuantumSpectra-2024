import jax.numpy as jnp
import jax_dataclasses as jdc
from jaxtyping import Float, Int, Array, Scalar

from quantumspectra_2024.modules.absorption import (
    AbsorptionModel as Model,
    AbsorptionSpectrum,
)


@jdc.pytree_dataclass(kw_only=True)
class StarkModel(Model):
    """A general model for Stark absorption spectra."""

    neutral_submodel: Model

    positive_field_strength: Float[Scalar, ""]
    positive_field_sum_percent: Float[Scalar, ""] = 0.5

    field_delta_dipole: Float[Scalar, ""]
    field_delta_polarizability: Float[Scalar, ""]

    def get_absorption(self) -> AbsorptionSpectrum:
        # get absorption spectrum sample energies (x values)
        neutral_submodel = self.get_neutral_submodel()
        positive_submodel = self.get_charged_submodel(field_strength_scalar=1.0)
        negative_submodel = self.get_charged_submodel(field_strength_scalar=-1.0)

        neutral_absorption = neutral_submodel.get_absorption()

        neutral_spectrum = neutral_absorption.intensities
        positive_spectrum = positive_submodel.get_absorption().intensities
        negative_spectrum = negative_submodel.get_absorption().intensities

        charged_half_sum = jnp.sum(
            jnp.array([positive_spectrum, negative_spectrum]), axis=0
        )

        electroabsorption_spectrum = charged_half_sum - jnp.array(neutral_spectrum)

        return AbsorptionSpectrum(
            energies=neutral_absorption.energies,
            intensities=electroabsorption_spectrum,
        )

    def get_neutral_submodel(self) -> Model:
        # replace neutral submodel with own point values
        neutral_submodel = jdc.replace(
            self.neutral_submodel,
            start_energy=self.start_energy,
            end_energy=self.end_energy,
            num_points=self.num_points,
        )

        return neutral_submodel

    def get_charged_submodel(self, field_strength_scalar: Float[Scalar, ""]) -> Model:
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
        raise NotImplementedError("StarkModel does not support apply_electric_field")
