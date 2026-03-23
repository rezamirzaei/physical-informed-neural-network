from __future__ import annotations

from physics_informed_neural_network import ProjectConfig
from physics_informed_neural_network.data import (
    generate_reference_solution,
    sample_boundary_collocation,
    sample_initial_collocation,
    sample_interior_collocation,
)


def test_reference_solution_and_collocation_shapes() -> None:
    config = ProjectConfig()
    config.data.nx = 32
    config.data.nt = 16

    reference = generate_reference_solution(config.pde, config.data)
    interior = sample_interior_collocation(config.pde, n=64, seed=1)
    boundary = sample_boundary_collocation(config.pde, n_per_edge=8, seed=2)
    initial = sample_initial_collocation(config.pde, n=10, seed=3)

    assert reference.u_array().shape == (16, 32)
    assert interior.shape == (64, 2)
    assert boundary.shape == (16, 2)
    assert initial.shape == (10, 2)
