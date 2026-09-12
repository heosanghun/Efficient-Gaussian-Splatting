"""Compact Neural & Spherical Appearance Models for 3D Gaussian Splatting
Common base class and activations based on Hahlbohm et al., 2026.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns the activation function phi(c_hat) as described in Section 3.1."""
    name_lower = name.lower()
    if name_lower == "relu":
        # Standard 3DGS ReLU formulation: ReLU(c_hat + 0.5)
        return lambda x: F.relu(x + 0.5)
    elif name_lower == "softplus":
        # Smooth rectifier: softplus(c_hat + 0.5, beta=10)
        return lambda x: F.softplus(x + 0.5, beta=10.0)
    elif name_lower == "sigmoid":
        # Scaled sigmoid: sigma(4 * c_hat)
        return lambda x: torch.sigmoid(4.0 * x)
    elif name_lower == "hard_sigmoid" or name_lower == "clamp":
        # Hard clamp: clamp(c_hat + 0.5, 0.0, 1.0)
        return lambda x: torch.clamp(x + 0.5, 0.0, 1.0)
    elif name_lower in ("none", "identity"):
        return lambda x: x
    else:
        raise ValueError(f"Unknown color activation: {name}")


def get_inverse_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Inverse activation to initialize base color from SfM point colors c_sfm in [0, 1]."""
    name_lower = name.lower()
    if name_lower in ("relu", "softplus", "hard_sigmoid", "clamp"):
        # c0 = c_sfm - 0.5
        return lambda c: c - 0.5
    elif name_lower == "sigmoid":
        # c0 = logit(c_sfm) / 4.0
        return lambda c: (torch.logit(torch.clamp(c, 1e-4, 1.0 - 1e-4))) / 4.0
    elif name_lower in ("none", "identity"):
        return lambda c: c
    else:
        raise ValueError(f"Unknown color activation: {name}")


class BaseAppearanceModel(nn.Module, ABC):
    """Abstract base class for all view-dependent appearance representations.
    
    Implements the unified formulation:
        c(d) = phi(c_0 + A_psi(d))
    where:
        c_0 is the base color (initialized from SfM),
        A_psi(d) is the view-dependent appearance residual,
        phi is the color activation.
    """

    def __init__(
        self,
        num_primitives: int,
        color_activation: str = "relu",
        residual_activation: str = "tanh",
        active_degree: int = 0,
        max_degree: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.num_primitives = num_primitives
        self.color_activation_name = color_activation
        self.residual_activation_name = residual_activation
        self.color_activation = get_activation(color_activation)
        self.inv_color_activation = get_inverse_activation(color_activation)
        self.active_degree = active_degree
        self.max_degree = max_degree
        self.device = torch.device(device)

        # Base color c_0 (N, 3), initialized to zero by default (corresponding to 0.5 gray)
        self.base_colors = nn.Parameter(
            torch.zeros((num_primitives, 3), dtype=torch.float32, device=self.device)
        )

    def initialize_from_rgb(self, rgb: torch.Tensor) -> None:
        """Initializes base colors from initial point cloud colors c_sfm in [0, 1]."""
        with torch.no_grad():
            self.base_colors.data.copy_(self.inv_color_activation(rgb.to(self.device)))

    @abstractmethod
    def compute_residual(self, directions: torch.Tensor) -> torch.Tensor:
        """Computes the appearance residual A_psi(d) for each primitive.
        
        Args:
            directions: Unit viewing directions (N, 3) or (B, N, 3) pointing from
                        camera to primitive or primitive to camera.
        Returns:
            Residual color tensor of shape (N, 3) or (B, N, 3).
        """
        pass

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Computes the final color c(d) = phi(c_0 + A_psi(d))."""
        residual = self.compute_residual(directions)
        if residual is None or self.active_degree == 0:
            # During initial steps (degree 0), only base color is used
            c_hat = self.base_colors
            if directions.dim() == 3:
                c_hat = c_hat.unsqueeze(0).expand(directions.shape[0], -1, -1)
        else:
            base = self.base_colors
            if directions.dim() == 3:
                base = base.unsqueeze(0).expand(directions.shape[0], -1, -1)
            c_hat = base + residual
        return self.color_activation(c_hat)

    def set_active_degree(self, degree: int) -> None:
        """Sets the active frequency / appearance degree for progressive training."""
        self.active_degree = min(degree, self.max_degree)

    def step_iteration(self, iteration: int, step_interval: int = 1000) -> None:
        """Progressively activates appearance degrees every `step_interval` iterations."""
        target_degree = min(self.max_degree, iteration // step_interval)
        if target_degree != self.active_degree:
            self.set_active_degree(target_degree)

    @abstractmethod
    def get_bytes_per_primitive(self) -> int:
        """Returns the memory footprint per primitive in bytes."""
        pass

    @abstractmethod
    def get_param_groups(self, lr_base: float, lr_residual: float) -> list:
        """Returns parameter groups with specific learning rates and optimizer settings."""
        pass
