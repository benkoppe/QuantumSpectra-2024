from quantumspectra_2024.modules.Config import (
    parse_config,
    initialize_absorption_from_config,
)
from quantumspectra_2024.modules.absorption import AbsorptionModel, AbsorptionSpectrum

from quantumspectra_2024.absorption.two_state import TwoStateModel


def main():
    config: dict = parse_config("Compute absorption spectrum with a given config file.")

    str_to_model: dict = {
        "two_state": TwoStateModel,
    }
    model: AbsorptionModel = initialize_absorption_from_config(config, str_to_model)

    spectrum: AbsorptionSpectrum = model.get_absorption()


if __name__ == "__main__":
    main()
