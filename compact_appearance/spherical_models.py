"""Spherical Appearance Models for 3D Gaussian Splatting
Implementations of SH, NASG, NASGabor, SV, and Diffuse-only based on Hahlbohm et al., 2026.
"""

import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from compact_appearance.appearance_base import BaseAppearanceModel
from compact_appearance.neural_appearance import evaluate_sh_basis


class SphericalHarmonicsAppearance(BaseAppearanceModel):
    """Standard 3DGS 3rd-Degree Spherical Harmonics Appearance Model.
    
    Parameters per primitive:
    - 3 base color components (Degree 0)
    - 45 residual SH coefficients (Degrees 1, 2, 3: 15 per RGB channel)
    - Total: 48 parameters = 192 bytes per primitive (FP32).
    """

    def __init__(
        self,
        num_primitives: int,
        max_degree: int = 3,
        color_activation: str = "relu",
        active_degree: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            num_primitives=num_primitives,
            color_activation=color_activation,
            active_degree=active_degree,
            max_degree=max_degree,
            device=device,
        )
        # Number of residual SH basis functions: (max_degree + 1)^2 - 1 = 16 - 1 = 15
        num_sh_bases = (max_degree + 1) ** 2 - 1
        # Coefficients shape: (N, 15, 3) initialized to zero
        self.sh_coefficients = nn.Parameter(
            torch.zeros((num_primitives, num_sh_bases, 3), dtype=torch.float32, device=self.device)
        )

    def compute_residual(self, directions: torch.Tensor) -> torch.Tensor:
        if self.active_degree == 0:
            if directions.dim() == 3:
                return torch.zeros((*directions.shape[:-1], 3), dtype=directions.dtype, device=directions.device)
            return torch.zeros((self.num_primitives, 3), dtype=directions.dtype, device=directions.device)

        # Evaluate SH basis (excluding degree 0 at index 0)
        sh_basis = evaluate_sh_basis(directions, active_degree=self.active_degree)[..., 1:]  # (*, 15)

        # Compute dot product over basis: sum_{i=1}^15 Y_i(d) * c_i
        # sh_coefficients: (N, 15, 3)
        if directions.dim() == 3:
            # directions: (B, N, 3), sh_basis: (B, N, 15)
            residual = torch.einsum("bni,nic->bnc", sh_basis, self.sh_coefficients)
        else:
            # directions: (N, 3), sh_basis: (N, 15)
            residual = torch.einsum("ni,nic->nc", sh_basis, self.sh_coefficients)

        return residual

    def get_bytes_per_primitive(self) -> int:
        """3 base + 45 residual = 48 float32 values = 192 bytes."""
        return (3 + self.sh_coefficients.shape[1] * 3) * 4

    def get_param_groups(self, lr_base: float = 0.008, lr_residual: float = 0.0025, **kwargs) -> List[dict]:
        return [
            {"params": [self.base_colors], "lr": lr_base, "name": "base_colors"},
            {"params": [self.sh_coefficients], "lr": lr_residual, "name": "sh_coefficients"},
        ]


class NASGAppearance(BaseAppearanceModel):
    """Normalized Anisotropic Spherical Gaussians (NASG).
    
    Each lobe i has:
    - Learned orientation axis (3 params)
    - Anisotropic angular extent / sharpness (2 params)
    - RGB amplitude a_i (3 params)
    Total: 8 parameters per lobe. For 1 lobe: 3 base + 8 lobe = 11 floats = 44 bytes.
    """

    def __init__(
        self,
        num_primitives: int,
        num_lobes: int = 1,
        color_activation: str = "relu",
        active_degree: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            num_primitives=num_primitives,
            color_activation=color_activation,
            active_degree=active_degree,
            max_degree=num_lobes,
            device=device,
        )
        self.num_lobes = num_lobes
        # Orientations (N, num_lobes, 3)
        self.lobe_axes = nn.Parameter(
            torch.randn((num_primitives, num_lobes, 3), dtype=torch.float32, device=self.device)
        )
        # Sharpness / log bandwidth (N, num_lobes, 2)
        self.lobe_sharpness = nn.Parameter(
            torch.ones((num_primitives, num_lobes, 2), dtype=torch.float32, device=self.device) * 2.0
        )
        # Amplitudes a_i (N, num_lobes, 3), initialized to zero
        self.lobe_amplitudes = nn.Parameter(
            torch.zeros((num_primitives, num_lobes, 3), dtype=torch.float32, device=self.device)
        )

    def compute_residual(self, directions: torch.Tensor) -> torch.Tensor:
        if self.active_degree == 0:
            if directions.dim() == 3:
                return torch.zeros((*directions.shape[:-1], 3), dtype=directions.dtype, device=directions.device)
            return torch.zeros((self.num_primitives, 3), dtype=directions.dtype, device=directions.device)

        norm_axes = F.normalize(self.lobe_axes, p=2, dim=-1)  # (N, L, 3)
        lambdas = torch.exp(torch.clamp(self.lobe_sharpness, -2.0, 6.0))  # (N, L, 2)

        if directions.dim() == 3:
            B = directions.shape[0]
            d = F.normalize(directions, p=2, dim=-1).unsqueeze(2)  # (B, N, 1, 3)
            axes = norm_axes.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, L, 3)
            cos_theta = (d * axes).sum(dim=-1, keepdim=True)  # (B, N, L, 1)
            dist_sq = torch.clamp(2.0 * (1.0 - cos_theta), min=0.0)
            avg_sharpness = lambdas.mean(dim=-1, keepdim=True).unsqueeze(0)  # (1, N, L, 1)
            weights = torch.exp(-avg_sharpness * dist_sq)  # (B, N, L, 1)
            amps = torch.tanh(self.lobe_amplitudes).unsqueeze(0)  # (1, N, L, 3)
            residual = (weights * amps).sum(dim=2)  # (B, N, 3)
        else:
            d = F.normalize(directions, p=2, dim=-1).unsqueeze(1)  # (N, 1, 3)
            cos_theta = (d * norm_axes).sum(dim=-1, keepdim=True)  # (N, L, 1)
            dist_sq = torch.clamp(2.0 * (1.0 - cos_theta), min=0.0)
            avg_sharpness = lambdas.mean(dim=-1, keepdim=True)  # (N, L, 1)
            weights = torch.exp(-avg_sharpness * dist_sq)  # (N, L, 1)
            amps = torch.tanh(self.lobe_amplitudes)  # (N, L, 3)
            residual = (weights * amps).sum(dim=1)  # (N, 3)

        return residual

    def get_bytes_per_primitive(self) -> int:
        """3 base + 8 per lobe = 11 floats = 44 bytes for 1 lobe."""
        return (3 + self.num_lobes * 8) * 4

    def get_param_groups(self, lr_base: float = 0.008, lr_residual: float = 0.003, **kwargs) -> List[dict]:
        return [
            {"params": [self.base_colors], "lr": lr_base, "name": "base_colors"},
            {"params": [self.lobe_axes, self.lobe_sharpness, self.lobe_amplitudes], "lr": lr_residual, "name": "lobe_params"},
        ]


class NASGaborAppearance(NASGAppearance):
    """NASG with Gabor Cosine Carrier.
    
    Adds 1 scalar frequency parameter per lobe:
        G_tilde(d) = G(d) * cos^2(omega * angle)
    Total: 3 base + 9 per lobe = 12 floats = 48 bytes for 1 lobe.
    """

    def __init__(
        self,
        num_primitives: int,
        num_lobes: int = 1,
        color_activation: str = "relu",
        active_degree: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            num_primitives=num_primitives,
            num_lobes=num_lobes,
            color_activation=color_activation,
            active_degree=active_degree,
            device=device,
        )
        # Learned carrier frequency per lobe (N, num_lobes, 1)
        self.lobe_frequencies = nn.Parameter(
            torch.zeros((num_primitives, num_lobes, 1), dtype=torch.float32, device=self.device)
        )

    def compute_residual(self, directions: torch.Tensor) -> torch.Tensor:
        if self.active_degree == 0:
            if directions.dim() == 3:
                return torch.zeros((*directions.shape[:-1], 3), dtype=directions.dtype, device=directions.device)
            return torch.zeros((self.num_primitives, 3), dtype=directions.dtype, device=directions.device)

        norm_axes = F.normalize(self.lobe_axes, p=2, dim=-1)
        lambdas = torch.exp(torch.clamp(self.lobe_sharpness, -2.0, 6.0))

        if directions.dim() == 3:
            B = directions.shape[0]
            d = F.normalize(directions, p=2, dim=-1).unsqueeze(2)
            axes = norm_axes.unsqueeze(0).expand(B, -1, -1, -1)
            cos_theta = torch.clamp((d * axes).sum(dim=-1, keepdim=True), -1.0, 1.0)
            theta = torch.acos(cos_theta)
            dist_sq = 2.0 * (1.0 - cos_theta)
            avg_sharpness = lambdas.mean(dim=-1, keepdim=True).unsqueeze(0)
            gauss_weights = torch.exp(-avg_sharpness * dist_sq)
            freq = self.lobe_frequencies.unsqueeze(0)
            carrier = torch.cos(freq * theta) ** 2
            amps = torch.tanh(self.lobe_amplitudes).unsqueeze(0)
            residual = (gauss_weights * carrier * amps).sum(dim=2)
        else:
            d = F.normalize(directions, p=2, dim=-1).unsqueeze(1)
            cos_theta = torch.clamp((d * norm_axes).sum(dim=-1, keepdim=True), -1.0, 1.0)
            theta = torch.acos(cos_theta)
            dist_sq = 2.0 * (1.0 - cos_theta)
            avg_sharpness = lambdas.mean(dim=-1, keepdim=True)
            gauss_weights = torch.exp(-avg_sharpness * dist_sq)
            carrier = torch.cos(self.lobe_frequencies * theta) ** 2
            amps = torch.tanh(self.lobe_amplitudes)
            residual = (gauss_weights * carrier * amps).sum(dim=1)

        return residual

    def get_bytes_per_primitive(self) -> int:
        """3 base + 9 per lobe = 12 floats = 48 bytes for 1 lobe."""
        return (3 + self.num_lobes * 9) * 4

    def get_param_groups(self, lr_base: float = 0.008, lr_residual: float = 0.003, **kwargs) -> List[dict]:
        groups = super().get_param_groups(lr_base=lr_base, lr_residual=lr_residual, **kwargs)
        groups[1]["params"].append(self.lobe_frequencies)
        return groups


class SphericalVoronoiAppearance(BaseAppearanceModel):
    """Spherical Voronoi (SV) Appearance Model.
    
    Soft interpolation over sites {s_i} on the sphere:
        A_psi(d) = sum_i softmax(-tau_i * dist(d, s_i)) * a_i
    Each site has orientation (3), sharpness tau (1), RGB (3) = 7 floats.
    For 7 sites: 3 base + 7 * 7 = 52 floats = 208 bytes.
    """

    def __init__(
        self,
        num_primitives: int,
        num_sites: int = 7,
        color_activation: str = "relu",
        active_degree: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            num_primitives=num_primitives,
            color_activation=color_activation,
            active_degree=active_degree,
            max_degree=1,
            device=device,
        )
        self.num_sites = num_sites
        # Sites positions (N, S, 3)
        self.site_positions = nn.Parameter(
            torch.randn((num_primitives, num_sites, 3), dtype=torch.float32, device=self.device)
        )
        # Sharpness tau (N, S, 1)
        self.site_sharpness = nn.Parameter(
            torch.ones((num_primitives, num_sites, 1), dtype=torch.float32, device=self.device) * 2.0
        )
        # Site RGB values (N, S, 3)
        self.site_colors = nn.Parameter(
            torch.zeros((num_primitives, num_sites, 3), dtype=torch.float32, device=self.device)
        )

    def compute_residual(self, directions: torch.Tensor) -> torch.Tensor:
        norm_sites = F.normalize(self.site_positions, p=2, dim=-1)
        tau = torch.exp(torch.clamp(self.site_sharpness, -2.0, 5.0))

        if directions.dim() == 3:
            B = directions.shape[0]
            d = F.normalize(directions, p=2, dim=-1).unsqueeze(2)  # (B, N, 1, 3)
            sites = norm_sites.unsqueeze(0).expand(B, -1, -1, -1)
            dist = torch.norm(d - sites, p=2, dim=-1, keepdim=True)  # (B, N, S, 1)
            logits = -tau.unsqueeze(0) * dist
            weights = F.softmax(logits, dim=2)
            colors = self.site_colors.unsqueeze(0)
            residual = (weights * colors).sum(dim=2)
        else:
            d = F.normalize(directions, p=2, dim=-1).unsqueeze(1)  # (N, 1, 3)
            dist = torch.norm(d - norm_sites, p=2, dim=-1, keepdim=True)  # (N, S, 1)
            logits = -tau * dist
            weights = F.softmax(logits, dim=1)
            residual = (weights * self.site_colors).sum(dim=1)

        return residual

    def get_bytes_per_primitive(self) -> int:
        """3 base + 7 sites * 7 floats = 52 floats = 208 bytes."""
        return (3 + self.num_sites * 7) * 4

    def get_param_groups(self, lr_base: float = 0.008, lr_residual: float = 0.003, **kwargs) -> List[dict]:
        return [
            {"params": [self.base_colors], "lr": lr_base, "name": "base_colors"},
            {"params": [self.site_positions, self.site_sharpness, self.site_colors], "lr": lr_residual, "name": "sv_params"},
        ]


class DiffuseOnlyAppearance(BaseAppearanceModel):
    """Diffuse-only (No view-dependent appearance / Degree 0 SH).
    
    12 bytes per primitive (3 floats for base color).
    """

    def __init__(
        self,
        num_primitives: int,
        color_activation: str = "relu",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__(
            num_primitives=num_primitives,
            color_activation=color_activation,
            active_degree=0,
            max_degree=0,
            device=device,
        )

    def compute_residual(self, directions: torch.Tensor) -> torch.Tensor:
        if directions.dim() == 3:
            return torch.zeros((*directions.shape[:-1], 3), dtype=directions.dtype, device=directions.device)
        return torch.zeros((self.num_primitives, 3), dtype=directions.dtype, device=directions.device)

    def get_bytes_per_primitive(self) -> int:
        return 3 * 4  # 12 bytes

    def get_param_groups(self, lr_base: float = 0.008, lr_ignored: float = 0.0) -> List[dict]:
        return [{"params": [self.base_colors], "lr": lr_base, "name": "base_colors"}]
