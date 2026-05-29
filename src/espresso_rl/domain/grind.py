from __future__ import annotations


class GrindNormalizer:
    """Convert grinder steps to microns and back."""

    def __init__(self, step_size_um: float) -> None:
        if step_size_um <= 0:
            raise ValueError("step_size_um must be positive")
        self.step_size_um = step_size_um

    def steps_to_um(self, steps: float) -> float:
        return steps * self.step_size_um

    def um_to_steps(self, um: float) -> int:
        return round(um / self.step_size_um)

    def snap_delta_um(self, delta_um: float) -> float:
        return self.um_to_steps(delta_um) * self.step_size_um

