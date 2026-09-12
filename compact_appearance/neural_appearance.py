"""Compact Neural Appearance Model for 3D Gaussian Splatting
Implementation based on Florian Hahlbohm et al., 2026.
(arXiv:2609.05255v1 / "Compact Neural Appearance Models for Efficient Gaussian Splatting")
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from compact_appearance.appearance_base import BaseAppearanceModel


# Spherical Harmonics constants
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def evaluate_sh_basis(directions: torch.Tensor, active_degree: int = 3) -> torch.Tensor:
    """Computes standard real Spherical Harmonics basis functions up to degree 3.
    
    Args:
        directions: Unit vector tensor (*, 3), typically [x, y, z].
        active_degree: Active maximum degree (0, 1, 2, or 3).
    Returns:
        Tensor of shape (*, 16) with SH basis values. Inactive bands are zeroed out.
    """
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    # Pre-allocate 16 basis components
    sh = torch.zeros((*directions.shape[:-1], 16), dtype=directions.dtype, device=directions.device)

    # Degree 0 (1 basis) - constant bias term
    sh[..., 0] = C0

    if active_degree >= 1:
        # Degree 1 (3 basis functions)
        sh[..., 1] = -C1 * y
        sh[..., 2] = C1 * z
        sh[..., 3] = -C1 * x

    if active_degree >= 2:
        # Degree 2 (5 basis functions)
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        sh[..., 4] = C2[0] * xy
        sh[..., 5] = C2[1] * yz
        sh[..., 6] = C2[2] * (2.0 * zz - xx - yy)
        sh[..., 7] = C2[3] * xz
        sh[..., 8] = C2[4] * (xx - yy)

    if active_degree >= 3:
        # Degree 3 (7 basis functions)
        sh[..., 9] = C3[0] * y * (3.0 * xx - yy)
        sh[..., 10] = C3[1] * xy * z
        sh[..., 11] = C3[2] * y * (4.0 * zz - xx - yy)
        sh[..., 12] = C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
        sh[..., 13] = C3[4] * x * (4.0 * zz - xx - yy)
        sh[..., 14] = C3[5] * z * (xx - yy)
        sh[..., 15] = C3[6] * x * (xx - 3.0 * yy)

    return sh


def frequency_encode(features: torch.Tensor, n_frequencies: int = 1) -> torch.Tensor:
    """Applies sinusoidal frequency encoding gamma_feat^k(f) = [sin(2^j f), cos(2^j f)].
    
    Args:
        features: Tensor of shape (*, F).
        n_frequencies: Number of frequencies k (default 1).
    Returns:
        Tensor of shape (*, F * 2 * n_frequencies).
    """
    if n_frequencies <= 0:
        return features
    encoded = []
    for j in range(n_frequencies):
        freq = (2.0 ** j) * features
        encoded.append(torch.sin(freq))
        encoded.append(torch.cos(freq))
    return torch.cat(encoded, dim=-1)


class SharedTinyMLP(nn.Module):
    """Tiny shared MLP decoder as specified in Section 4 and Appendix A.4.
    
    Architecture:
    - Input dimension: 32 (16 SH direction basis + 16 encoded latent features)
    - 2 hidden layers with 16 neurons each
    - ReLU activations
    - Bias: None (bias is provided by the constant degree-0 SH basis component)
    - Output layer: 3 dimensions (RGB residual)
    - Zero-initialized final weight matrix (guarantees zero residual at start)
    - Total parameters across entire scene: 32*16 + 16*16 + 16*3 = 816 parameters.
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 16,
        out_dim: int = 3,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        output_layer = nn.Linear(hidden_dim, out_dim, bias=False)
        # Initialize output weights to zero to ensure exact zero initial residual
        nn.init.zeros_(output_layer.weight)
        layers.append(output_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactNeuralAppearance(BaseAppearanceModel):
    """Official Compact Neural Appearance Model for 3D Gaussian Splatting.
    
    Encodes view-dependent appearance using:
    - 8 latent features f in R^8 per primitive (stored in FP16 / 16 bytes).
    - 3 base color components c_0 in R^3 per primitive (stored in FP32 / 12 bytes).
    - Total per-primitive footprint = 28 bytes (vs 192 bytes for degree-3 SH).
    - Shared tiny MLP with 816 scene-level parameters.
    """

    def __init__(
        self,
        num_primitives: int,
        feature_dim: int = 8,
        n_frequencies: int = 1,
        color_activation: str = "relu",
        residual_activation: str = "tanh",
        active_degree: int = 0,
        max_degree: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            num_primitives=num_primitives,
            color_activation=color_activation,
            residual_activation=residual_activation,
            active_degree=active_degree,
            max_degree=max_degree,
            device=device,
        )
        self.feature_dim = feature_dim
        self.n_frequencies = n_frequencies

        # Latent feature vector f in R^F per primitive, initialized to zero
        self.latent_features = nn.Parameter(
            torch.zeros((num_primitives, feature_dim), dtype=torch.float32, device=self.device)
        )

        # Network input dimension: 16 (SH basis 0..3) + 16 (frequency encoded 8 features) = 32
        n_encoded_features = feature_dim * 2 * n_frequencies
        n_sh_dims = 16
        self.in_dim = n_encoded_features + n_sh_dims
        assert self.in_dim == 32, f"Expected input dimension 32 for Tensor Core alignment, got {self.in_dim}"

        # Shared Tiny MLP
        self.shared_mlp = SharedTinyMLP(
            in_dim=self.in_dim,
            hidden_dim=16,
            out_dim=3,
            num_hidden_layers=2,
        ).to(self.device)

    def compute_residual(self, directions: torch.Tensor) -> torch.Tensor:
        """Computes the neural view-dependent appearance residual A_psi(d).
        
        Args:
            directions: Unit direction tensor of shape (N, 3) or (B, N, 3).
        Returns:
            Residual color tensor A_psi(d) = tanh(MLP(x)).
        """
        if self.active_degree == 0:
            # During initial phase, residual is completely disabled (returns zeros)
            if directions.dim() == 3:
                return torch.zeros((*directions.shape[:-1], 3), dtype=directions.dtype, device=directions.device)
            return torch.zeros((self.num_primitives, 3), dtype=directions.dtype, device=directions.device)

        # 1. Evaluate SH direction basis gamma_dir^B(d)
        sh_basis = evaluate_sh_basis(directions, active_degree=self.active_degree)  # (*, 16)

        # 2. Encode latent features gamma_feat^k(f)
        encoded_feats = frequency_encode(self.latent_features, self.n_frequencies)  # (N, 16)

        if directions.dim() == 3:
            # Broadcast over batch dimension B (e.g. multiple camera views)
            B = directions.shape[0]
            encoded_feats = encoded_feats.unsqueeze(0).expand(B, -1, -1)

        # 3. Concatenate to assemble 32-dim input x = [sh_basis, encoded_features]
        mlp_input = torch.cat([sh_basis, encoded_feats], dim=-1)  # (*, 32)

        # 4. Evaluate tiny shared MLP
        raw_residual = self.shared_mlp(mlp_input)  # (*, 3)

        # 5. Apply bounded residual activation tanh
        if self.residual_activation_name == "tanh":
            return torch.tanh(raw_residual)
        elif self.residual_activation_name in ("none", "identity"):
            return raw_residual
        else:
            return torch.tanh(raw_residual)

    def get_bytes_per_primitive(self) -> int:
        """Returns deployed footprint: 12B (FP32 base color) + 16B (FP16 latent features) = 28 Bytes."""
        # Base colors: 3 * 4 = 12 bytes
        # Latent features: 8 * 2 = 16 bytes (stored/deployed as FP16)
        return 12 + self.feature_dim * 2

    def get_param_groups(
        self,
        lr_base: float = 0.008,
        lr_residual: float = 0.005,
        lr_mlp: float = 0.001,
        **kwargs,
    ) -> List[dict]:
        """Configures parameter groups as described in Section 4 & Appendix B.
        
        Shared MLP weights use Adam beta2=0.99 because they receive dense gradients
        from all visible primitives in every iteration.
        """
        return [
            {"params": [self.base_colors], "lr": lr_base, "name": "base_colors"},
            {"params": [self.latent_features], "lr": lr_residual, "name": "latent_features"},
            {
                "params": list(self.shared_mlp.parameters()),
                "lr": lr_mlp,
                "betas": (0.9, 0.99),
                "name": "mlp_weights",
            },
        ]
