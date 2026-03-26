"""Comprehensive tests for shared utility functions."""

from __future__ import annotations

import numpy as np
import torch

from physics_informed_neural_network.utils import (
    ensure_directory,
    latin_hypercube_sample,
    select_device,
    set_global_seed,
)


class TestEnsureDirectory:
    """Verify directory creation helper."""

    def test_creates_missing_directory(self, tmp_path) -> None:
        target = tmp_path / "a" / "b" / "c"
        assert not target.exists()
        result = ensure_directory(target)
        assert target.is_dir()
        assert result == target

    def test_returns_existing_directory(self, tmp_path) -> None:
        result = ensure_directory(tmp_path)
        assert result == tmp_path

    def test_idempotent(self, tmp_path) -> None:
        target = tmp_path / "subdir"
        ensure_directory(target)
        ensure_directory(target)
        assert target.is_dir()


class TestSetGlobalSeed:
    """Verify reproducibility seeding."""

    def test_numpy_reproducibility(self) -> None:
        set_global_seed(42)
        a = np.random.rand(10)
        set_global_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_torch_reproducibility(self) -> None:
        set_global_seed(42)
        a = torch.randn(10)
        set_global_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_different_seeds_give_different_values(self) -> None:
        set_global_seed(1)
        a = np.random.rand(100)
        set_global_seed(2)
        b = np.random.rand(100)
        assert not np.allclose(a, b)


class TestSelectDevice:
    """Verify device selection logic."""

    def test_cpu_explicit(self) -> None:
        device = select_device("cpu")
        assert device.type == "cpu"

    def test_auto_returns_valid_device(self) -> None:
        device = select_device("auto")
        assert device.type in ("cpu", "cuda", "mps")

    def test_cuda_fallback(self) -> None:
        device = select_device("cuda")
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"

    def test_mps_fallback(self) -> None:
        device = select_device("mps")
        if torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"


class TestLatinHypercubeSample:
    """Verify Latin Hypercube Sampling."""

    def test_shape(self) -> None:
        samples = latin_hypercube_sample(50, bounds=[(-1.0, 1.0), (0.0, 1.0)], seed=0)
        assert samples.shape == (50, 2)

    def test_within_bounds(self) -> None:
        bounds = [(-2.0, 3.0), (0.0, 10.0)]
        samples = latin_hypercube_sample(100, bounds=bounds, seed=7)
        assert np.all(samples[:, 0] >= -2.0) and np.all(samples[:, 0] <= 3.0)
        assert np.all(samples[:, 1] >= 0.0) and np.all(samples[:, 1] <= 10.0)

    def test_reproducibility(self) -> None:
        a = latin_hypercube_sample(20, bounds=[(0.0, 1.0)], seed=42)
        b = latin_hypercube_sample(20, bounds=[(0.0, 1.0)], seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self) -> None:
        a = latin_hypercube_sample(30, bounds=[(0.0, 1.0)], seed=1)
        b = latin_hypercube_sample(30, bounds=[(0.0, 1.0)], seed=2)
        assert not np.allclose(a, b)

    def test_3d(self) -> None:
        samples = latin_hypercube_sample(25, bounds=[(-1.0, 1.0), (0.0, 2.0), (5.0, 10.0)], seed=3)
        assert samples.shape == (25, 3)
        assert np.all(samples[:, 2] >= 5.0) and np.all(samples[:, 2] <= 10.0)


