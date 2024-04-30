import matplotlib.pyplot as plt

from quantumspectra_2024.models.three_state.ThreeStateSimpleModel import (
    ThreeStateSimpleModel,
)

model = ThreeStateSimpleModel(
    temperature_kelvin=300,
    broadening=200,
    le_energy_gap=12358,
    ct_energy_gap=10745,
    gs_ct_coupling=100,
    ct_le_coupling=850,
    d_LE=1.0,
    d_CT=1.0,
    le_mode_couplings=[-0.85, -2.8],
    ct_mode_couplings=[0.85, 4.0],
    mode_basis_sets=[20, 200],
    mode_frequencies=[1400, 100],
)

spectrum = model.get_absorption()

plt.plot(spectrum.energies, spectrum.intensities)
plt.show()
