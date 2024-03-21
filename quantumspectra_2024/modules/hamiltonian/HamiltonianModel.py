import jax_dataclasses as jdc
from jaxtyping import Float, Int, Scalar, Array


@jdc.pytree_dataclass
class HamiltonianModel:
    """A hamiltonian model for a quantum system"""

    basis_sets: jdc.Static[Int[Array, "num_states"]]

    transfer_integral: Float[Scalar, ""]
    state_energies: Float[Array, "num_states"]

    mode_frequencies: Float[Array, "num_modes"]
    mode_couplings: Float[Array, "num_states num_modes"]
