"""
Observation encoder: (5, 100) shot profile + 3 scalars → 256-dim embedding.

Profile channels (axis 0):
    0  pressure         actual pump pressure (bar)
    1  target_pressure  profile-prescribed target (bar)
    2  flow             actual pump flow (ml/s)
    3  target_flow      profile-prescribed target (ml/s)
    4  weight           scale output weight (g)

Scalar features:
    grind_um / 5000       (typical range 0–5000 μm → roughly 0–1)
    dose_g   / 25         (typical range 15–20 g   → roughly 0–1)
    step_size_um / 20     (typical range 5–15 μm   → roughly 0–1)
"""

import torch
import torch.nn as nn

# Fixed normalisation constants (baked in so the network sees ~unit-scale inputs)
_PROFILE_SCALE = torch.tensor(
    [9.0, 9.0, 5.0, 5.0, 40.0], dtype=torch.float32
)  # bar, bar, ml/s, ml/s, g

EMBED_DIM = 256


class ObservationEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()

        # 1D-CNN over the 100-point time axis
        # Input : (B, 5, 100)
        # Output: (B, 128, 13) → flatten → 1664 → Linear(embed_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(5,  32,  kernel_size=5, stride=2, padding=2),  # → (B, 32, 50)
            nn.ELU(),
            nn.Conv1d(32, 64,  kernel_size=5, stride=2, padding=2),  # → (B, 64, 25)
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # → (B, 128, 13)
            nn.ELU(),
        )
        cnn_out_dim = 128 * 13  # 1664

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, embed_dim),
            nn.ELU(),
        )

        # Small MLP for scalar context
        self.scalar_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + 32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ELU(),
        )

        # Register profile scale as a buffer (moves to device with the module)
        self.register_buffer("profile_scale", _PROFILE_SCALE)

    def forward(
        self,
        profile: torch.Tensor,     # (*, 5, 100)
        grind_um: torch.Tensor,    # (*,)
        dose_g: torch.Tensor,      # (*,)
        step_size_um: torch.Tensor,  # (*,)
    ) -> torch.Tensor:             # (*, embed_dim)

        *batch, C, T = profile.shape
        B = 1
        for d in batch:
            B *= d

        # Normalise profile channels
        scale = self.profile_scale.view(1, 5, 1)  # broadcast over batch & time
        x = profile.view(B, C, T) / scale

        # CNN
        cnn_feat = self.cnn(x)                     # (B, 128, 13)
        cnn_feat = cnn_feat.flatten(1)             # (B, 1664)
        cnn_emb  = self.cnn_proj(cnn_feat)         # (B, embed_dim)

        # Scalar features
        scalars = torch.stack([
            grind_um.view(B)    / 5000.0,
            dose_g.view(B)      / 25.0,
            step_size_um.view(B) / 20.0,
        ], dim=-1)                                  # (B, 3)
        scalar_emb = self.scalar_mlp(scalars)       # (B, 32)

        # Fuse
        emb = self.fusion(torch.cat([cnn_emb, scalar_emb], dim=-1))  # (B, embed_dim)
        return emb.view(*batch, EMBED_DIM)
