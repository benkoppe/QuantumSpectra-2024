import multiprocessing
import numpy as np

from quantumspectra_2024.models import TwoStateModel

from tests.validation_data.fortran import TWO_STATE_DATA, TWO_STATE_STATICS

num_cpus = 5
data_comparison_generator = TWO_STATE_DATA.models_vs_fortran_ydata(
    TwoStateModel, TWO_STATE_STATICS
)

num_close = 0
total = 0

for new_ydata, fortran_ydata in data_comparison_generator:
    if np.allclose(new_ydata, fortran_ydata):
        num_close += 1
    else:
        percentage_close = np.sum(np.isclose(new_ydata, fortran_ydata)) / len(new_ydata)
        print(f"Percentage close: {percentage_close:.2f}")
    total += 1

print(f"Number of close results: {num_close}")
print(f"Total number of results: {total}")
print(f"Percentage of close results: {num_close / total * 100:.2f}%")
