"""Comprehensive tests for PINN data generation and collocation sampling."""

from __future__ import annotations

import numpy as np

from physics_informed_neural_network.config import PDEConfig, ProjectConfig
from physics_informed_neural_network.data import (
    evaluate_reference_solution,
    generate_reference_solution,
    sample_boundary_collocation,
    sample_initial_collocation,
    sample_interior_collocation,
    sample_observations,
)


# ---------------------------------------------------------------------------
# Reference solution
# ---------------------------------------------------------------------------


class TestReferenceSolution:
    """Verify analytical Burgers solution generation."""

    def test_grid_shapes(self) -> None:
        config = ProjectConfig()
        config.data.nx = 32
        config.data.nt = 16
        ref = generate_reference_solution(config.pde, config.data)
        assert ref.u_array().shape == (16, 32)
        assert len(ref.x) == 32
        assert len(ref.t) == 16

    def test_initial_condition(self) -> None:
        """u(x, 0) should match -sin(pi x)."""
        pde = PDEConfig()
        x = np.linspace(pde.x_min, pde.x_max, 64)
        t = np.array([0.0])
        u = evaluate_reference_solution(x, t, pde.viscosity)
        expected = -np.sin(np.pi * x)
        np.testing.assert_allclose(u[0], expected, atol=1e-4)

    def test_boundary_conditions(self) -> None:
        """u(-1, t) and u(1, t) should be approximately zero."""
        pde = PDEConfig()
        x = np.array([pde.x_min, pde.x_max])
        t = np.linspace(pde.t_min, pde.t_max, 20)
        u = evaluate_reference_solution(x, t, pde.viscosity)
        np.testing.assert_allclose(u[:, 0], 0.0, atol=1e-4)
        np.testing.assert_allclose(u[:, 1], 0.0, atol=1e-4)

    def test_solution_bounded(self) -> None:
        """For the standard problem, |u| should not exceed ~1."""
        pde = PDEConfig()
        x = np.linspace(pde.x_min, pde.x_max, 64)
        t = np.linspace(pde.t_min, pde.t_max, 32)
        u = evaluate_reference_solution(x, t, pde.viscosity)
        assert np.all(np.abs(u) < 2.0)

    def test_meshgrid(self) -> None:
        config = ProjectConfig()
        config.data.nx = 10
        config.data.nt = 5
        ref = generate_reference_solution(config.pde, config.data)
        X, T = ref.meshgrid()
        assert X.shape == (5, 10)
        assert T.shape == (5, 10)

    def test_viscosity_stored(self) -> None:
        config = ProjectConfig()
        ref = generate_reference_solution(config.pde, config.data)
        assert ref.viscosity == config.pde.viscosity


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


class TestObservations:
    """Verify observation sampling."""

    def test_observation_count(self) -> None:
        config = ProjectConfig()
        config.data.nx = 32
        config.data.nt = 16
        ref = generate_reference_solution(config.pde, config.data)
        obs = sample_observations(ref, n_points=50, noise_std=0.0, seed=42)
        assert obs.n_points == 50
        assert len(obs.x) == 50
        assert len(obs.t) == 50
        assert len(obs.u) == 50

    def test_noiseless_observations_match_reference(self) -> None:
        config = ProjectConfig()
        config.data.nx = 32
        config.data.nt = 16
        ref = generate_reference_solution(config.pde, config.data)
        obs = sample_observations(ref, n_points=100, noise_std=0.0, seed=42)
        u_arr = ref.u_array()
        for x_val, t_val, u_val in zip(obs.x, obs.t, obs.u):
            xi = np.argmin(np.abs(np.array(ref.x) - x_val))
            ti = np.argmin(np.abs(np.array(ref.t) - t_val))
            assert abs(u_val - u_arr[ti, xi]) < 1e-10

    def test_noisy_observations_differ(self) -> None:
        config = ProjectConfig()
        config.data.nx = 32
        config.data.nt = 16
        ref = generate_reference_solution(config.pde, config.data)
        obs = sample_observations(ref, n_points=200, noise_std=0.1, seed=42)
        # With noise, at least some observations should differ from the clean reference
        u_arr = ref.u_array()
        diffs = []
        for x_val, t_val, u_val in zip(obs.x, obs.t, obs.u):
            xi = np.argmin(np.abs(np.array(ref.x) - x_val))
            ti = np.argmin(np.abs(np.array(ref.t) - t_val))
            diffs.append(abs(u_val - u_arr[ti, xi]))
        assert max(diffs) > 0.001

    def test_reproducibility(self) -> None:
        config = ProjectConfig()
        config.data.nx = 32
        config.data.nt = 16
        ref = generate_reference_solution(config.pde, config.data)
        obs1 = sample_observations(ref, n_points=50, noise_std=0.01, seed=7)
        obs2 = sample_observations(ref, n_points=50, noise_std=0.01, seed=7)
        np.testing.assert_array_equal(obs1.u, obs2.u)


# ---------------------------------------------------------------------------
# Collocation samplers
# ---------------------------------------------------------------------------


class TestCollocationSampling:
    """Verify collocation point samplers."""

    def test_interior_shape(self) -> None:
        pde = PDEConfig()
        pts = sample_interior_collocation(pde, n=100, seed=0)
        assert pts.shape == (100, 2)

    def test_interior_within_bounds(self) -> None:
        pde = PDEConfig()
        pts = sample_interior_collocation(pde, n=200, seed=1)
        assert np.all(pts[:, 0] >= pde.x_min) and np.all(pts[:, 0] <= pde.x_max)
        assert np.all(pts[:, 1] >= pde.t_min) and np.all(pts[:, 1] <= pde.t_max)

    def test_boundary_shape(self) -> None:
        pde = PDEConfig()
        pts = sample_boundary_collocation(pde, n_per_edge=10, seed=2)
        assert pts.shape == (20, 2)  # 10 left + 10 right

    def test_boundary_on_edges(self) -> None:
        pde = PDEConfig()
        pts = sample_boundary_collocation(pde, n_per_edge=20, seed=3)
        x_vals = pts[:, 0]
        # All should be either x_min or x_max
        on_left = np.isclose(x_vals, pde.x_min)
        on_right = np.isclose(x_vals, pde.x_max)
        assert np.all(on_left | on_right)

    def test_initial_shape(self) -> None:
        pde = PDEConfig()
        pts = sample_initial_collocation(pde, n=50, seed=4)
        assert pts.shape == (50, 2)

    def test_initial_at_t_min(self) -> None:
        pde = PDEConfig()
        pts = sample_initial_collocation(pde, n=30, seed=5)
        np.testing.assert_allclose(pts[:, 1], pde.t_min)

    def test_initial_x_within_bounds(self) -> None:
        pde = PDEConfig()
        pts = sample_initial_collocation(pde, n=100, seed=6)
        assert np.all(pts[:, 0] >= pde.x_min) and np.all(pts[:, 0] <= pde.x_max)
