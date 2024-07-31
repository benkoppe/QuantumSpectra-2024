import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

parent = Path(__file__).parent

fortran_data = np.loadtxt(parent / "TwoState_absorption_1.dat")

plt.plot(fortran_data[:, 0], fortran_data[:, 1], label="col 1")

plt.show()
