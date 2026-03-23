from __future__ import annotations

import pytest
from pydantic import ValidationError

from physics_informed_neural_network import ProjectConfig


def test_project_config_rejects_invalid_spatial_bounds() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(pde={"x_min": 1.0, "x_max": -1.0})
