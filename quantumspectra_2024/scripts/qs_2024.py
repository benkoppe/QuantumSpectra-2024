from quantumspectra_2024.common.Config import (
    parse_config,
    initialize_absorption_from_config,
    save_spectrum_from_config,
)
from quantumspectra_2024.common.absorption import (
    AbsorptionModel,
    AbsorptionSpectrum,
)

from quantumspectra_2024.models import TwoStateModel, MLJModel, StarkModel


def main():
    config: dict = parse_config("Compute absorption spectrum with a given config file.")

    str_to_model: dict = {
        "two_state": TwoStateModel,
        "mlj": MLJModel,
        "stark": StarkModel,
    }
    model: AbsorptionModel = initialize_absorption_from_config(
        config=config, str_to_model=str_to_model
    )

    spectrum: AbsorptionSpectrum = model.get_absorption()

    save_spectrum_from_config(config=config, spectrum=spectrum)


if __name__ == "__main__":
    main()
