from typing import Type
from argparse import ArgumentParser
import tomllib

from quantumspectra_2024.modules.absorption import AbsorptionModel

CONFIG_ARG_NAME = "config_path"
CONFIG_ARG_HELP = "Path to the configuration file."

OUT_CONFIG_NAME = "out"
OUT_REQUIRED_KEYS = ["filename", "data", "plot", "overwrite"]

MODEL_CONFIG_NAME = "model"
MODEL_NAME_KEY = "name"


def initialize_absorption_from_config(
    config: dict, str_to_model: dict[str, Type[AbsorptionModel]]
) -> AbsorptionModel:
    model_config = config[MODEL_CONFIG_NAME]
    model_name = model_config[MODEL_NAME_KEY]

    if model_name not in str_to_model:
        raise ValueError(
            f"Config model name '{model_name}' not recognized. "
            f"Available models: {','.join(str_to_model.keys())}"
        )

    model = str_to_model[model_name]

    model_config.pop(MODEL_NAME_KEY)
    return model(**model_config)


def parse_config(program_description: str) -> dict:
    parser = ArgumentParser(description=program_description)

    parser.add_argument(CONFIG_ARG_NAME, type=str, help=CONFIG_ARG_HELP)
    config_path = parser.parse_args().config_path

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    validate_config(data)

    return data


def validate_config(data: dict) -> None:
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
    for key in keys:
        if key not in data:
            raise ValueError(
                f"Required config key '{key}' not found in {key_type} data."
            )
