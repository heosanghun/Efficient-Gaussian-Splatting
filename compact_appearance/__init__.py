"""Compact Neural & Spherical Appearance Models for 3D Gaussian Splatting
Reference implementation of Florian Hahlbohm et al., 2026.
"""

from compact_appearance.appearance_base import (
    BaseAppearanceModel,
    get_activation,
    get_inverse_activation,
)
from compact_appearance.neural_appearance import (
    CompactNeuralAppearance,
    SharedTinyMLP,
    evaluate_sh_basis,
    frequency_encode,
)
from compact_appearance.spherical_models import (
    SphericalHarmonicsAppearance,
    NASGAppearance,
    NASGaborAppearance,
    SphericalVoronoiAppearance,
    DiffuseOnlyAppearance,
)


def create_appearance_model(
    model_type: str,
    num_primitives: int,
    color_activation: str = "relu",
    residual_activation: str = "tanh",
    device: str = "cuda",
    **kwargs,
) -> BaseAppearanceModel:
    """Factory function for creating appearance models as analyzed in the paper."""
    model_type_upper = model_type.upper()
    if model_type_upper in ("NEURAL", "OURS"):
        return CompactNeuralAppearance(
            num_primitives=num_primitives,
            color_activation=color_activation,
            residual_activation=residual_activation,
            device=device,
            **kwargs,
        )
    elif model_type_upper == "SH":
        return SphericalHarmonicsAppearance(
            num_primitives=num_primitives,
            color_activation=color_activation,
            device=device,
            **kwargs,
        )
    elif model_type_upper == "NASG":
        return NASGAppearance(
            num_primitives=num_primitives,
            color_activation=color_activation,
            device=device,
            **kwargs,
        )
    elif model_type_upper == "NASGABOR":
        return NASGaborAppearance(
            num_primitives=num_primitives,
            color_activation=color_activation,
            device=device,
            **kwargs,
        )
    elif model_type_upper == "SV":
        return SphericalVoronoiAppearance(
            num_primitives=num_primitives,
            color_activation=color_activation,
            device=device,
            **kwargs,
        )
    elif model_type_upper in ("NONE", "DIFFUSE"):
        return DiffuseOnlyAppearance(
            num_primitives=num_primitives,
            color_activation=color_activation,
            device=device,
        )
    else:
        raise ValueError(f"Unknown appearance model type: {model_type}")


__all__ = [
    "BaseAppearanceModel",
    "CompactNeuralAppearance",
    "SharedTinyMLP",
    "SphericalHarmonicsAppearance",
    "NASGAppearance",
    "NASGaborAppearance",
    "SphericalVoronoiAppearance",
    "DiffuseOnlyAppearance",
    "create_appearance_model",
    "evaluate_sh_basis",
    "frequency_encode",
    "get_activation",
    "get_inverse_activation",
]
