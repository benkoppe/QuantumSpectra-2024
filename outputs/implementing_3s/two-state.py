from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parent = Path(__file__).parent

from quantumspectra_2024.models import TwoStateModel

model = TwoStateModel(
    temperature_kelvin=300,
    energy_gap=10745,
    transfer_integral=100,
    mode_basis_sets=[20, 200],
    mode_frequencies=[1400, 100],
    mode_couplings=[0.85, 4.0],
)

abs = model.get_absorption()

plt.plot(abs.energies, abs.intensities)
plt.show()
