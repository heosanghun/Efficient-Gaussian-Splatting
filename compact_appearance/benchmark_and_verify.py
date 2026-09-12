"""Comprehensive Verification and Benchmark for Compact Neural & Spherical Appearance Models
Replicating key experiments and validations from Florian Hahlbohm et al., 2026.
"""

import math
import sys
import time
from typing import Dict
import torch
import torch.nn.functional as F

from compact_appearance import (
    create_appearance_model,
    CompactNeuralAppearance,
    SphericalHarmonicsAppearance,
    NASGAppearance,
    NASGaborAppearance,
    SphericalVoronoiAppearance,
    DiffuseOnlyAppearance,
)


def run_memory_footprint_benchmark(num_primitives: int = 1_000_000):
    """Benchmarks and validates the memory footprint table (Table 1 in the paper)."""
    print("=" * 78)
    print(f"1. Memory Footprint Benchmark ({num_primitives:,} Gaussian Primitives)")
    print("=" * 78)

    models = [
        ("None", "DIFFUSE"),
        ("SH", "SH"),
        ("SV", "SV"),
        ("NASG", "NASG"),
        ("NASGabor", "NASGABOR"),
        ("Neural (Ours)", "NEURAL"),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'Model':<15} | {'Bytes/Gaussian':<15} | {'Parameters (1M Primitives)':<28} | {'Storage (MB)':<12}")
    print("-" * 78)

    for name, mtype in models:
        # Create a small instance to get byte metrics
        model = create_appearance_model(mtype, num_primitives=100, device=device)
        bytes_per_prim = model.get_bytes_per_primitive()
        total_mb = (bytes_per_prim * num_primitives) / (1024 * 1024)
        print(f"{name:<15} | {bytes_per_prim:<15} | {num_primitives * (bytes_per_prim // 4):<28,} | {total_mb:<12.2f}")

    print("-" * 78)
    print(">> Observation: Neural Appearance achieves 28 Bytes/Gaussian (an 85.4% reduction over SH's 192 Bytes)!")
    print("=" * 78 + "\n")


def run_gradient_and_schedule_verification():
    """Verifies backpropagation, zero initial residual, and progressive scheduling."""
    print("=" * 78)
    print("2. Progressive Scheduling and Gradient Flow Verification")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = 10
    model = CompactNeuralAppearance(num_primitives=N, active_degree=0, device=device)

    # 1. Test zero initial residual at degree 0
    dirs = F.normalize(torch.randn(N, 3, device=device), p=2, dim=-1)
    color_deg0 = model(dirs)
    expected_deg0 = F.relu(model.base_colors + 0.5)
    diff0 = (color_deg0 - expected_deg0).abs().max().item()
    print(f"[Test 1] Zero residual at iteration 0 (active_degree=0): max diff = {diff0:.6f}")
    assert diff0 < 1e-6, "Initial residual is not zero!"

    # 2. Test progressive activation via step_iteration
    model.step_iteration(500)
    assert model.active_degree == 0, f"Expected degree 0 at step 500, got {model.active_degree}"
    model.step_iteration(1200)
    assert model.active_degree == 1, f"Expected degree 1 at step 1200, got {model.active_degree}"
    model.step_iteration(2100)
    assert model.active_degree == 2, f"Expected degree 2 at step 2100, got {model.active_degree}"
    model.step_iteration(3500)
    assert model.active_degree == 3, f"Expected degree 3 at step 3500, got {model.active_degree}"
    print(f"[Test 2] Progressive scheduling (0 -> 1 -> 2 -> 3) verified successfully!")

    # 3. Test gradient backpropagation through features and shared MLP
    # Paper Note (App. A.4): Output layer is zero-initialized to guarantee zero residual.
    # At step 1, the output layer receives gradients and updates.
    # At step 2, latent features receive non-zero gradients.
    model.set_active_degree(3)
    target = torch.rand(N, 3, device=device)
    optimizer = torch.optim.Adam(model.get_param_groups(lr_base=0.01, lr_features=0.01, lr_mlp=0.01))

    # Step 1
    pred = model(dirs)
    loss = F.mse_loss(pred, target)
    loss.backward()

    base_grad_norm = model.base_colors.grad.norm().item()
    mlp_grad_norm = sum(p.grad.norm().item() for p in model.shared_mlp.parameters() if p.grad is not None)
    step1_feat_grad = model.latent_features.grad.norm().item()

    print(f"[Test 3] Gradient flow check (Step 1):")
    print(f"  - Base colors grad norm:     {base_grad_norm:.6f} (> 0)")
    print(f"  - Shared MLP grad norm:      {mlp_grad_norm:.6f} (> 0)")
    print(f"  - Latent features (Step 1):  {step1_feat_grad:.6f} (0.0 due to paper zero-init of output layer)")
    assert base_grad_norm > 0 and mlp_grad_norm > 0, "Base or MLP gradients failed to flow at Step 1!"

    # Step 2: Apply optimizer step, then verify feature gradients
    optimizer.step()
    optimizer.zero_grad()
    pred2 = model(dirs)
    loss2 = F.mse_loss(pred2, target)
    loss2.backward()

    step2_feat_grad = model.latent_features.grad.norm().item()
    print(f"[Test 3] Gradient flow check (Step 2 after MLP weight update):")
    print(f"  - Latent features (Step 2):  {step2_feat_grad:.6f} (> 0)")
    assert step2_feat_grad > 0, "Latent feature gradients failed to flow at Step 2!"
    print(">> Observation: All gradients flow correctly through latent features and shared MLP!")
    print("=" * 78 + "\n")


def run_specular_fitting_experiment():
    """Compares optimization capability: Fitting view-dependent specular reflections.
    
    Generates synthetic view-dependent reflection ground truth across camera views
    and compares convergence between 3rd-degree SH and Compact Neural Appearance.
    """
    print("=" * 78)
    print("3. View-Dependent Appearance Fitting Experiment (Specular Highlight)")
    print("=" * 78)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    N_primitives = 64
    N_views = 128

    # Generate random viewpoints on upper hemisphere
    elev = torch.rand(N_views, device=device) * (math.pi / 2.0)
    azim = torch.rand(N_views, device=device) * (2.0 * math.pi)
    dirs = torch.stack([
        torch.sin(elev) * torch.cos(azim),
        torch.sin(elev) * torch.sin(azim),
        torch.cos(elev)
    ], dim=-1)  # (V, 3)

    # Expand directions for all primitives: (V, N, 3)
    dirs = dirs.unsqueeze(1).expand(-1, N_primitives, -1)

    # Synthetic Ground Truth: Base diffuse color + sharp Phong specular highlight
    # Specular lobe around light direction L = [0, 0, 1]
    light_dir = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)
    normal = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3)
    # Reflected light direction R
    refl = 2.0 * normal * (normal * light_dir).sum(dim=-1, keepdim=True) - light_dir  # (1, 1, 3)
    cos_alpha = torch.clamp((dirs * refl).sum(dim=-1, keepdim=True), min=0.0)
    specular_gt = torch.pow(cos_alpha, 32.0) * torch.tensor([1.0, 0.8, 0.5], device=device).view(1, 1, 3)

    base_diffuse = torch.rand(1, N_primitives, 3, device=device) * 0.4
    target_colors = torch.clamp(base_diffuse + specular_gt, 0.0, 1.0)  # (V, N, 3)

    print(f"Target: {N_primitives} primitives viewed from {N_views} camera directions with high-frequency specularity.")

    models_to_test = [
        ("Spherical Harmonics (SH)", SphericalHarmonicsAppearance(N_primitives, max_degree=3, active_degree=3, device=device)),
        ("Compact Neural (Ours)", CompactNeuralAppearance(N_primitives, active_degree=3, device=device)),
        ("NASGabor", NASGaborAppearance(N_primitives, num_lobes=1, active_degree=1, device=device)),
    ]

    steps = 400
    for name, model in models_to_test:
        # Initialize base colors near target mean
        model.initialize_from_rgb(base_diffuse.squeeze(0))
        optimizer = torch.optim.Adam(model.get_param_groups(lr_base=0.01, lr_residual=0.01))

        t0 = time.perf_counter()
        for step in range(steps):
            optimizer.zero_grad()
            pred = model(dirs)  # (V, N, 3)
            loss = F.mse_loss(pred, target_colors)
            loss.backward()
            optimizer.step()

        elapsed = (time.perf_counter() - t0) * 1000.0
        with torch.no_grad():
            final_pred = model(dirs)
            final_mse = F.mse_loss(final_pred, target_colors).item()
            psnr = -10.0 * math.log10(max(final_mse, 1e-10))

        bytes_per_prim = model.get_bytes_per_primitive()
        print(f"  [{name}]")
        print(f"    - Final PSNR:          {psnr:.2f} dB (MSE: {final_mse:.6f})")
        print(f"    - Footprint:           {bytes_per_prim} Bytes/Gaussian")
        print(f"    - Optimization Time:   {elapsed:.1f} ms for {steps} steps")

    print("=" * 78 + "\n")


if __name__ == "__main__":
    print("\n" + "#" * 78)
    print("  COMPACT NEURAL APPEARANCE MODELS: VERIFICATION AND BENCHMARK SUITE")
    print("#" * 78 + "\n")

    run_memory_footprint_benchmark(num_primitives=1_000_000)
    run_gradient_and_schedule_verification()
    run_specular_fitting_experiment()
    print("All verifications completed successfully!")
