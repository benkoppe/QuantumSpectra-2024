import matplotlib.pyplot as plt
import numpy as np

from quantumspectra_2024.models import TwoStateModel
from quantumspectra_2024.models.three_state.ThreeStateModel import ThreeStateModel

two_model = TwoStateModel(
    temperature_kelvin=300,
    energy_gap=10745,
    transfer_integral=100,
    mode_basis_sets=[5, 10],
    mode_frequencies=[1400, 100],
    mode_couplings=[0.85, 4.0],
)

three_model = ThreeStateModel(
    temperature_kelvin=300,
    ct_energy_gap=10745,
    le_energy_gap=12358,
    gs_ct_coupling=100,
    ct_le_coupling=0,
    ct_mode_couplings=[0.85, 4.0],
    le_mode_couplings=[0, 0],
    mode_basis_sets=[5, 10],
    mode_frequencies=[1400, 100],
)

two_model_h = two_model.get_hamiltonian()
three_model_h = three_model.get_hamiltonian()

two_matrix = two_model_h.get_matrix()
three_matrix = three_model_h.get_matrix()

print(np.allclose(two_matrix, three_matrix[:100, :100]))

two_model_vals, two_model_vects = two_model_h.get_diagonalization()
three_model_vals, three_model_vects = three_model_h.get_diagonalization()

print(np.allclose(two_model_vals, three_model_vals[:100]))
