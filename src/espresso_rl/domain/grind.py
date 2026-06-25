from __future__ import annotations


class GrindNormalizer:
    """Convert grinder steps to microns and back."""

    def __init__(self, microns_per_step: float) -> None:
        if microns_per_step <= 0:
            raise ValueError("microns_per_step must be positive")
        self.microns_per_step = microns_per_step

    def steps_to_um(self, steps: float) -> float:
        return steps * self.microns_per_step

    def um_to_steps(self, um: float) -> int:
        return round(um / self.microns_per_step)

    def snap_delta_um(self, delta_um: float) -> float:
        return self.um_to_steps(delta_um) * self.microns_per_step

