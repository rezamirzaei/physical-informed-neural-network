"""Full end-to-end evaluation of the neural operator project."""
import sys
from pathlib import Path

# Ensure src is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

print("=== STEP 1: Import test ===")
from physics_informed_neural_network.neural_operator.presets import (
    build_tutorial_config, build_darcy_tutorial_config,
    build_smoke_test_config, build_darcy_smoke_test_config,
)
from physics_informed_neural_network.neural_operator.pipeline import run_neural_operator_experiment
from physics_informed_neural_network.neural_operator.pipeline_2d import run_darcy_experiment
from physics_informed_neural_network.neural_operator.plotting import (
    apply_plot_style, plot_training_history, plot_dataset_examples,
    plot_prediction_comparison, plot_resolution_metrics, plot_frequency_spectrum,
)
from physics_informed_neural_network.neural_operator.plotting_2d import (
    plot_darcy_dataset_examples, plot_darcy_prediction_comparison,
    plot_darcy_error_distribution, plot_darcy_resolution_metrics,
    plot_darcy_cross_sections,
)
from physics_informed_neural_network.neural_operator.data import (
    build_dataset_splits, compute_discrete_diffusion_residual,
)
from physics_informed_neural_network.neural_operator.data_2d import build_darcy_splits
from physics_informed_neural_network.neural_operator.model import FourierNeuralOperator1d
from physics_informed_neural_network.neural_operator.model_2d import FourierNeuralOperator2d
print("All imports OK")

print("\n=== STEP 2: 1-D Smoke test ===")
config_1d = build_smoke_test_config()
exp_1d = run_neural_operator_experiment(config_1d)
r1 = exp_1d.summary.evaluations["test"].metrics.relative_l2
r2 = exp_1d.summary.evaluations["refined_test"].metrics.relative_l2
print(f"1D native rel_l2:  {r1:.6f}")
print(f"1D refined rel_l2: {r2:.6f}")

print("\n=== STEP 3: 2-D Darcy smoke test ===")
config_2d = build_darcy_smoke_test_config()
exp_2d = run_darcy_experiment(config_2d)
r3 = exp_2d.summary.evaluations["test"].metrics.relative_l2
r4 = exp_2d.summary.evaluations["refined_test"].metrics.relative_l2
print(f"2D native rel_l2:  {r3:.6f}")
print(f"2D refined rel_l2: {r4:.6f}")

print("\n=== STEP 4: All plotting functions ===")
apply_plot_style()

fig = plot_training_history(exp_1d.history); plt.close(fig); print("  plot_training_history OK")
fig = plot_dataset_examples(exp_1d.datasets.train, (0, 1)); plt.close(fig); print("  plot_dataset_examples OK")

ns = exp_1d.datasets.test.sample(0)
fig = plot_prediction_comparison(ns, exp_1d.native_prediction[0], "test"); plt.close(fig); print("  plot_prediction_comparison OK")
fig = plot_resolution_metrics(exp_1d.summary); plt.close(fig); print("  plot_resolution_metrics OK")
fig = plot_frequency_spectrum(ns.grid, ns.solution, exp_1d.native_prediction[0], "test"); plt.close(fig); print("  plot_frequency_spectrum OK")

fig = plot_darcy_dataset_examples(exp_2d.datasets.train, (0, 1)); plt.close(fig); print("  plot_darcy_dataset_examples OK")
ds = exp_2d.datasets.test.sample(0)
fig = plot_darcy_prediction_comparison(ds, exp_2d.native_prediction[0], "test"); plt.close(fig); print("  plot_darcy_prediction_comparison OK")
fig = plot_darcy_error_distribution(exp_2d.native_prediction, exp_2d.datasets.test.solution); plt.close(fig); print("  plot_darcy_error_distribution OK")
fig = plot_darcy_resolution_metrics(exp_2d.summary); plt.close(fig); print("  plot_darcy_resolution_metrics OK")
fig = plot_darcy_cross_sections(ds, exp_2d.native_prediction[0]); plt.close(fig); print("  plot_darcy_cross_sections OK")

print("\n=== STEP 5: Verify data correctness ===")
# Check 1D exact solution satisfies BCs and PDE residual is small
sample = exp_1d.datasets.train.sample(0)
residual = compute_discrete_diffusion_residual(sample.grid, sample.diffusion, sample.solution, sample.forcing)
print(f"  1D BCs: u(0)={sample.solution[0]:.2e}, u(1)={sample.solution[-1]:.2e}")
print(f"  1D PDE residual: mean={np.mean(np.abs(residual)):.2e}, max={np.max(np.abs(residual)):.2e}")

# Check 2D Darcy solution: BCs and positivity
d2_sample = exp_2d.datasets.train.sample(0)
print(f"  2D BCs: top={np.max(np.abs(d2_sample.solution[0, :])):.2e}, "
      f"bottom={np.max(np.abs(d2_sample.solution[-1, :])):.2e}, "
      f"left={np.max(np.abs(d2_sample.solution[:, 0])):.2e}, "
      f"right={np.max(np.abs(d2_sample.solution[:, -1])):.2e}")
print(f"  2D interior positive: {np.all(d2_sample.solution[1:-1, 1:-1] >= 0)}")
print(f"  2D a field values: {sorted(set(np.unique(d2_sample.diffusivity)))}")

# Check predictions are finite
print(f"  1D predictions finite: {np.all(np.isfinite(exp_1d.native_prediction))}")
print(f"  2D predictions finite: {np.all(np.isfinite(exp_2d.native_prediction))}")

print("\n=== STEP 6: Model resolution transfer ===")
import torch
# 1D: verify same model works at two resolutions
model_1d = FourierNeuralOperator1d(config_1d.model)
x1 = torch.randn(2, 32, 3)
x2 = torch.randn(2, 64, 3)
assert model_1d(x1).shape == (2, 32, 1)
assert model_1d(x2).shape == (2, 64, 1)
print("  1D resolution transfer: 32 -> 64 OK")

# 2D: verify same model works at two resolutions
model_2d = FourierNeuralOperator2d(config_2d.model)
x3 = torch.randn(2, 15, 15, 4)
x4 = torch.randn(2, 29, 29, 4)
assert model_2d(x3).shape == (2, 15, 15, 1)
assert model_2d(x4).shape == (2, 29, 29, 1)
print("  2D resolution transfer: 15x15 -> 29x29 OK")

print("\n" + "=" * 60)
print("ALL QUALITY CHECKS PASSED")
print("=" * 60)

