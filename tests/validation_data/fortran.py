import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class FortranValidationData:
    xdata: np.ndarray

    _param_ydata_mapping: dict[tuple, np.ndarray]

    # create a generator that yields the keys (as a dict) and values of the param_ydata_mapping
    def params_to_ydata(self):
        for param_tuples, ydata in self._param_ydata_mapping.items():
            # convert param tuples to a dict
            param_dict = {
                param_tuple[0]: param_tuple[1] for param_tuple in param_tuples
            }
            # convert tuple values in param_dict to lists
            param_dict = {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in param_dict.items()
            }
            # convert values to floats
            param_dict = {
                key: float(value) if isinstance(value, str) else value
                for key, value in param_dict.items()
            }
            # convert strings in lists to floats
            param_dict = {
                key: (
                    [float(v) if isinstance(v, str) else v for v in value]
                    if isinstance(value, list)
                    else value
                )
                for key, value in param_dict.items()
            }

            yield param_dict, ydata

    def models_vs_fortran_ydata(self, model_type, static_params):
        for params, fortran_ydata in self.params_to_ydata():
            model = model_type(**params, **static_params)
            spectra = model.get_absorption()
            new_ydata = np.array(spectra.intensities)

            # scale fortran ydata to match peak of new ydata
            source_peak = np.max(fortran_ydata)
            target_peak = np.max(new_ydata)

            scaling_factor = target_peak / source_peak

            scaled_fortran_ydata = fortran_ydata * scaling_factor

            yield new_ydata, scaled_fortran_ydata

    @classmethod
    def from_pickle(cls, path) -> "FortranValidationData":
        with open(path, "rb") as f:
            return pickle.load(f)


parent_dir = Path(__file__).parent

TWO_STATE_DATA = FortranValidationData.from_pickle(parent_dir / "two_state_data.pickle")
TWO_STATE_STATICS = {
    "mode_basis_sets": [20, 50],
}
