import multiprocessing
import numpy as np

from quantumspectra_2024.models import TwoStateModel

from validation_data.fortran import TWO_STATE_DATA, TWO_STATE_STATICS

num_cpus = 5
data_comparison_generator = TWO_STATE_DATA.models_vs_fortran_ydata(
    TwoStateModel, TWO_STATE_STATICS
)

for new_ydata, fortran_ydata in data_comparison_generator:
    print(np.allclose(new_ydata, fortran_ydata))
