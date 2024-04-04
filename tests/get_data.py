from pathlib import Path

start_dir = Path("/Users/ben/Documents/Research/11_Python/march-2023-fortran/merge")
target_filename = "2_states_absorption_1.dat"

param_mapping = {
    "AE22": "energy_gap",
    "T": "temperature_kelvin",
    "r2": "broadening",
    "t0": "transfer_integral",
    "g": "mode_couplings",
    "f": "mode_frequencies",
}

all_params_dict = {}

for file_path in start_dir.rglob(target_filename):
    relative_path = file_path.relative_to(start_dir)

    params_tuples = tuple(
        [tuple(part.split("=")) for part in relative_path.parent.parts]
    )
    all_params_dict[params_tuples] = file_path

print(len(all_params_dict))

# remove instances where t0=0
all_params_dict = {
    key: value for key, value in all_params_dict.items() if key[4][1] != "0.0"
}

# remove F, aa, and tdm222 from the keys
all_params_dict = {
    tuple([part for part in key if part[0] not in ["F", "aa", "tdm222"]]): value
    for key, value in all_params_dict.items()
}


def combine_two_keys(dict, key1, key2, conjoined_name):
    new_dict = {
        tuple(
            [part for part in key if part[0] not in [key1, key2]]
            + [
                (
                    conjoined_name,
                    tuple([part[1] for part in key if part[0] in [key1, key2]]),
                )
            ]
        ): value
        for key, value in dict.items()
    }
    return new_dict


# combine g1 and g2 into a single key with a tuple of values
all_params_dict = combine_two_keys(all_params_dict, "g1", "g2", "g")
# do the same with f1 and f2
all_params_dict = combine_two_keys(all_params_dict, "f1", "f2", "f")

# replace the keys with the actual parameter names
all_params_dict = {
    tuple(
        [
            (param_mapping[part[0]], part[1])
            for part in key
            if part[0] in param_mapping.keys()
        ]
    ): value
    for key, value in all_params_dict.items()
}

keys = list(all_params_dict.keys())

print(keys[0])

import numpy as np
from dataclasses import dataclass
import pickle
from validation_data.fortran import FortranValidationData


# load data at every file path

xdata = np.loadtxt(all_params_dict[keys[0]], usecols=(0,))

# convert each file path to ydata
param_ydata_mapping = {
    key: np.loadtxt(value, usecols=(1,)) for key, value in all_params_dict.items()
}

two_state_data = FortranValidationData(xdata, param_ydata_mapping)

# pickle the class
with open("two_state_data.pickle", "wb") as f:
    pickle.dump(two_state_data, f)
