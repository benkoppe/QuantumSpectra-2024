import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import random
import time
import numpy as np

from quantumspectra_2024.models.three_state.ThreeStateSimpleModel import (
    ThreeStateSimpleModel,
)

from quantumspectra_2024.models.two_state.TwoStateModel import TwoStateModel
from quantumspectra_2024.models.two_state.TwoStateComputation import compute_peaks

from quantumspectra_2024.common.hamiltonian.HamiltonianModel import HamiltonianModel

# model = ThreeStateSimpleModel(
#     temperature_kelvin=300,
#     broadening=200,
#     le_energy_gap=12358,
#     ct_energy_gap=10745,
#     gs_ct_coupling=100,
#     ct_le_coupling=850,
#     d_LE=1.0,
#     d_CT=1.0,
#     le_mode_couplings=[-0.85, -2.8],
#     ct_mode_couplings=[0.85, 4.0],
#     mode_basis_sets=[20, 200],
#     mode_frequencies=[1400, 100],
# )

# spectrum = model.get_absorption()

# plt.plot(spectrum.energies, spectrum.intensities)
# plt.show()

jax.config.update("jax_platform_name", "gpu")
# jax.config.update("jax_enable_x64", True)


def time_two_state():
    t = TwoStateModel(
        temperature_kelvin=random.choice([0.0, 300.0]),
        broadening=200,
        transfer_integral=random.randint(90, 110),
        energy_gap=8_000.0,
        mode_basis_sets=jnp.array([20, 200]),
        mode_frequencies=jnp.array([1200.0, 100.0]),
        mode_couplings=jnp.array([0.7, 2.0]),
    )

    start = time.time()

    t.get_absorption()

    end = time.time()

    return end - start


def time_matrix():
    h = HamiltonianModel(
        transfer_integrals=random.randint(90, 110),
        state_energies=jnp.array([0.0, 10_000.0]),
        mode_basis_sets=jnp.array([20, 200]),
        mode_localities=jnp.array([True, True]),
        mode_frequencies=jnp.array([1400, 100]),
        mode_state_couplings=jnp.array([[0.0, 0.7], [0.0, 1.6]]),
    )

    start = time.time()

    h.get_matrix()

    end = time.time()

    return end - start


N = 10
runtimes = []

for _ in range(N):
    runtime = time_two_state()
    print(runtime)
    runtimes.append(runtime)


print(f"Average runtime: {np.mean(runtimes):.2f} seconds")
print(f"Average runtime excluidng first: {np.mean(runtimes[1:]):.2f} seconds")
