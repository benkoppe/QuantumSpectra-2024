import os
import shutil
from typing import Type, Callable
from pathlib import Path
from argparse import ArgumentParser
import tomllib

from quantumspectra_2024.common.absorption import (
    AbsorptionModel,
    AbsorptionSpectrum,
)

SAMPLE_CONFIGS_SOURCE = "sample_configs/"

CONFIG_ARG_NAME = "config_path"
CONFIG_ARG_HELP = "Path to the configuration file."

OUT_CONFIG_NAME = "out"
OUT_REQUIRED_KEYS = ["filename", "data", "plot", "overwrite"]

MODEL_CONFIG_NAME = "model"
MODEL_NAME_KEY = "name"


def save_spectrum_from_config(config: dict, spectrum: AbsorptionSpectrum) -> None:
    """Saves an absorption spectrum to requested files based on a configuration dict.

    Currently configured to save absorption spectrum data and plot to CSV and PNG files, respectively.

    Parameters
    ----------
    config : dict
        the configuration dict.
    spectrum : AbsorptionSpectrum
        the absorption spectrum to save.
    """
    out_data = config[OUT_CONFIG_NAME]
    overwrite = out_data["overwrite"]

    if out_data["data"]:
        # save absorption spectrum data
        save_file(
            filename=f"{out_data['filename']}.csv",
            overwrite=overwrite,
            save_func=lambda fname: spectrum.save_data(fname),
        )

    if out_data["plot"]:
        # save absorption spectrum plot
        save_file(
            filename=f"{out_data['filename']}.png",
            overwrite=overwrite,
            save_func=lambda fname: spectrum.save_plot(fname),
        )


def save_file(filename: str, overwrite: bool, save_func: Callable[[str], None]) -> None:
    """Checks if a file can be saved and then saves it using a save function.

    Parameters
    ----------
    filename : str
        the filename to save to.
    overwrite : bool
        whether to overwrite the file if it already exists.
    save_func : Callable[[str], None]
        The function to use to save the file.
        Function expects a single string argument containing the path to save to.

    Raises
    ------
    ValueError
        if the parent directory of the save file does not exist.
    ValueError
        if the save file already exists and overwrite is set to False.
    """
    file = Path(filename)

    if not file.parent.exists():
        raise ValueError(
            f"Invalid save file: parent directory '{file.parent}' does not exist."
        )
    if file.exists() and not overwrite:
        raise ValueError(
            f"Save file '{file}' already exists and overwrite is set to False."
        )

    save_func(str(file))


def initialize_absorption_from_config(
    config: dict, str_to_model: dict[str, Type[AbsorptionModel]]
) -> AbsorptionModel:
    """Initializes an `AbsorptionModel` subclass from a configuration dict.

    Configuration dicts must have a `model` key with a `name` key specifying the model name.
    The model name must be a key in the `str_to_model` dict.

    Model subdicts are treated as submodels and are recursively initialized.
    This is useful for models that have submodels as attributes, such as `StarkModel`.

    Parameters
    ----------
    config : dict
        the configuration dict.
    str_to_model dict[str, Type[AbsorptionModel]]
        a dict mapping model names to model classes.

    Raises
    ------
    ValueError
        if the model name is not found in `str_to_model`.

    Returns
    -------
    AbsorptionModel
        the initialized absorption model.
    """
    model_config = config[MODEL_CONFIG_NAME]
    model_name = model_config[MODEL_NAME_KEY]

    if model_name not in str_to_model:
        raise ValueError(
            f"Config model name '{model_name}' not recognized. "
            f"Available models: {','.join(str_to_model.keys())}"
        )

    model = str_to_model[model_name]

    model_config.pop(MODEL_NAME_KEY)

    # replace subdicts with submodels
    for key, value in model_config.items():
        if isinstance(value, dict):
            submodel = initialize_absorption_from_config(
                config={MODEL_CONFIG_NAME: value, OUT_CONFIG_NAME: {}},
                str_to_model=str_to_model,
            )
            model_config[key] = submodel

    return model(**model_config)


def parse_config(program_description: str) -> dict:
    """Parse and validate a configuration dict for absorption spectrum computation from script arguments.

    Parameters
    ----------
    program_description : str
        the program description for the argument parser.

    Returns
    -------
    dict
        the absorption spectrum computation configuration dict."""
    parser = ArgumentParser(description=program_description)

    parser.add_argument(CONFIG_ARG_NAME, type=str, help=CONFIG_ARG_HELP)
    config_path = parser.parse_args().config_path

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    validate_config(data)

    return data


def validate_config(data: dict) -> None:
    """Validates an absorption spectrum computation configuration dict.

    Parameters
    ----------
    data : dict
        the configuration dict to validate.

    Raises
    ------
    ValueError
        if the config is not valid.
    """
    # check for required keys
    required_keys = [MODEL_CONFIG_NAME, OUT_CONFIG_NAME]
    ensure_keys_included(data, required_keys, "main")

    # check for required output keys
    out_data = data[OUT_CONFIG_NAME]
    ensure_keys_included(out_data, OUT_REQUIRED_KEYS, OUT_CONFIG_NAME)

    # check for required model keys
    model_data = data[MODEL_CONFIG_NAME]
    model_required_keys = [MODEL_NAME_KEY]
    ensure_keys_included(model_data, model_required_keys, MODEL_CONFIG_NAME)


def ensure_keys_included(data: dict, keys: list, key_type: str) -> None:
    """Ensures that a list of required keys are included in a data dict.

    Parameters
    ----------
    data : dict
        the data dict to check.
    keys : list
        the list of required keys.
    key_type : str
        the type of data dict being checked.

    Raises
    ------
    ValueError
        if a required key is not found in the data dict.
    """
    for key in keys:
        if key not in data:
            raise ValueError(
                f"Required config key '{key}' not found in {key_type} data."
            )


def copy_sample_configs(destination_folder: str, overwrite: bool = False):
    """Copies all sample configuration files to the given destination folder.

    Parameters
    ----------
    destination_folder : str
        the destination folder to copy the sample configs to.
    """
    # Ensure the destination folder path exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all files in the source folder
    source_folder = Path(SAMPLE_CONFIGS_SOURCE)
    for file in source_folder.iterdir():
        # Copy each file to the destination folder
        if file.is_file():
            destination_file = Path(destination_folder) / file.name
            if overwrite or not destination_file.exists():
                shutil.copy(file, destination_file)
            else:
                print(
                    f"File {destination_file} already exists, and overwriting is disabled. Skipping."
                )
