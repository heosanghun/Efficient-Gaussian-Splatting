import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import trimesh

WORKSPACE = Path(r"e:\00000000000000000_D드라이브 파일 이동\AI\Compact Neural")
sys.path.append(str(WORKSPACE))
sys.path.append(str(WORKSPACE / "models" / "TripoSR"))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
import rembg
from backend.ply_to_ngsplat import pack_to_ngsplat

_MODEL = None
_REMBG_SESSION = None

def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def get_model():
    """Lazy-loads and caches the TripoSR model on GPU."""
    global _MODEL
    if _MODEL is None:
        device = get_device()
        print(f"Loading TripoSR onto {device}...")
        t0 = time.time()
        _MODEL = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        _MODEL.renderer.set_chunk_size(8192)
        _MODEL.to(device)
        print(f"TripoSR loaded in {time.time() - t0:.2f}s onto {device}")
    return _MODEL

def get_rembg_session():
    """Lazy-loads and caches the rembg session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        _REMBG_SESSION = rembg.new_session()
    return _REMBG_SESSION

def generate_local_3dgs(
    image_path: str,
    output_ngsplat_path: str,
    num_splats: int = 60000,
    foreground_ratio: float = 0.85,
    resolution: int = 256,
) -> dict:
    """Generates 3D Gaussian Splatting scene (.ngsplat) directly on RTX 4090."""
    t0 = time.time()
    device = get_device()
    model = get_model()

    raw_img = Image.open(image_path).convert("RGBA")
    img_np = np.array(raw_img)
    has_alpha = raw_img.mode == "RGBA" and (img_np[:, :, 3].min() < 240)

    if has_alpha:
        img = resize_foreground(raw_img, foreground_ratio)
    else:
        session = get_rembg_session()
        img = remove_background(raw_img, session)
        img = resize_foreground(img, foreground_ratio)

    img_arr = np.array(img).astype(np.float32) / 255.0
    img_arr = img_arr[:, :, :3] * img_arr[:, :, 3:4] + (1 - img_arr[:, :, 3:4]) * 0.5
    input_img = Image.fromarray((img_arr * 255.0).astype(np.uint8))

    t_inf = time.time()
    with torch.no_grad():
        scene_codes = model([input_img], device=device)
    inference_sec = time.time() - t_inf

    t_mesh = time.time()
    meshes = model.extract_mesh(scene_codes, True, resolution=resolution)
    mesh = meshes[0]
    mesh_sec = time.time() - t_mesh

    target_splats = min(max(num_splats, len(mesh.vertices)), 80000)
    samples, face_indices = trimesh.sample.sample_surface(mesh, count=target_splats)
    normals = mesh.face_normals[face_indices]

    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        face_vertices = mesh.faces[face_indices]
        v_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
        colors = v_colors[face_vertices].mean(axis=1)
    else:
        colors = np.full((target_splats, 3), 0.7, dtype=np.float32)

    surface_area = mesh.area
    avg_spacing = np.sqrt(max(surface_area / target_splats, 1e-6)) * 1.3
    log_spacing = np.log(max(avg_spacing, 1e-5))

    scales = np.zeros((target_splats, 3), dtype=np.float32)
    scales[:, 0] = log_spacing
    scales[:, 1] = log_spacing
    scales[:, 2] = log_spacing - 1.2

    opacities = np.full(target_splats, 0.95, dtype=np.float32)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cross_prod = np.cross(z_axis, normals)
    dot_prod = normals[:, 2]

    rotations = np.zeros((target_splats, 4), dtype=np.float32)
    rotations[:, 0] = 1.0 + dot_prod
    rotations[:, 1] = cross_prod[:, 0]
    rotations[:, 2] = cross_prod[:, 1]
    rotations[:, 3] = cross_prod[:, 2]
    norm_q = np.linalg.norm(rotations, axis=-1, keepdims=True)
    norm_q[norm_q == 0] = 1.0
    rotations /= norm_q

    pack_to_ngsplat(
        out_path=output_ngsplat_path,
        positions=samples.astype(np.float32),
        colors=colors.astype(np.float32),
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        model_type="neural",
    )

    total_sec = time.time() - t0
    file_size_mb = os.path.getsize(output_ngsplat_path) / (1024 * 1024)

    return {
        "success": True,
        "num_splats": target_splats,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "file_size_mb": round(file_size_mb, 2),
        "inference_sec": round(inference_sec, 2),
        "mesh_sec": round(mesh_sec, 2),
        "total_sec": round(total_sec, 2),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
