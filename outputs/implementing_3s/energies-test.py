from pathlib import Path

parent = Path(__file__).parent

from quantumspectra_2024.models.three_state.ThreeStateSimpleModel import (
    ThreeStateSimpleModel,
)
from quantumspectra_2024.models.three_state.ThreeStateComputation import *

from quantumspectra_2024.common.hamiltonian import HamiltonianModel


class ModThreeStateSimpleModel(ThreeStateSimpleModel):
    def get_hamiltonian(self) -> HamiltonianModel:
        return HamiltonianModel(
            transfer_integrals=self.ct_le_coupling,
            state_energies=jnp.array([0, self.le_energy_gap - self.ct_energy_gap]),
            mode_basis_sets=jnp.array(self.mode_basis_sets),
            mode_localities=jnp.array([True, True]),
            mode_frequencies=jnp.array(self.mode_frequencies),
            mode_state_couplings=jnp.array(
                [
                    [ct_mode_coupling, le_mode_coupling]
                    for ct_mode_coupling, le_mode_coupling in zip(
                        self.ct_mode_couplings, self.le_mode_couplings
                    )
                ]
            ),
        )


model = ThreeStateSimpleModel(
    temperature_kelvin=0,
    ct_energy_gap=10745,
    le_energy_gap=12358,
    gs_ct_coupling=100,
    ct_le_coupling=850,
    d_LE=0,
    d_CT=0,
    ct_mode_couplings=[0.85, 4.0],
    le_mode_couplings=[-0.85, -2.8],
    mode_basis_sets=[10, 100],
    mode_frequencies=[1400, 100],
)

# otherModel = ModThreeStateSimpleModel(
#     temperature_kelvin=0,
#     ct_energy_gap=10745,
#     le_energy_gap=12358,
#     gs_ct_coupling=100,
#     ct_le_coupling=850,
#     d_LE=0,
#     d_CT=0,
#     ct_mode_couplings=[0.85, 4.0],
#     le_mode_couplings=[-0.85, -2.8],
#     mode_basis_sets=[10, 100],
#     mode_frequencies=[1400, 100],
# )


hamiltonian = model.get_hamiltonian()
# otherHamiltonian = otherModel.get_hamiltonian()

vals, vects = hamiltonian.get_diagonalization()
# otherVals, otherVects = otherHamiltonian.get_diagonalization()

# comparison = [
#     12451.170337782183,
#     13016.605175964234,
#     13312.866431663897,
#     13795.898386849098,
#     13899.533488181765,
#     14006.875809330428,
#     14974.252936961531,
#     15221.171474049006,
#     15465.754028473209,
#     16146.150070223353,
#     16270.944678636246,
#     16525.284625727596,
#     17873.568079009023,
#     18344.857131587843,
#     18633.686622491667,
#     19183.768431317763,
#     19276.016510681813,
# ]

import numpy as np

comparison = np.fromfile(parent / "Eigenvalues.dat", sep="\n")
comparison = comparison[1000:]

print(vals.shape)

print(vals)
print(comparison)

diffs = vals[:-100] - jnp.array(comparison)
# otherDiffs = (otherVals[:-100] + otherModel.ct_energy_gap) - jnp.array(comparison)

print(diffs.mean())
# print(otherDiffs.mean())

# print(diffs.mean())
# print(diffs.max())
# print(diffs.min())
# print(diffs.std())
