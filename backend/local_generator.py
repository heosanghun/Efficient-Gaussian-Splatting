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

from backend.ply_to_ngsplat import pack_to_ngsplat

_MODEL = None
_REMBG_SESSION = None

def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def get_model():
    """Lazy-loads and caches the TripoSR model on GPU."""
    global _MODEL
    if _MODEL is None:
        from tsr.system import TSR
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
        import rembg
        _REMBG_SESSION = rembg.new_session(model_name="u2net")
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
    from tsr.utils import remove_background, resize_foreground
    model = get_model()

    raw_img = Image.open(image_path).convert("RGBA")
    if max(raw_img.size) > 512:
        raw_img.thumbnail((512, 512), Image.Resampling.LANCZOS)

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

    # Center mesh at origin and normalize scale to radius 0.85
    mesh.vertices -= mesh.bounding_box.centroid
    max_dist = np.linalg.norm(mesh.vertices, axis=-1).max()
    if max_dist > 1e-4:
        mesh.vertices /= (max_dist / 0.85)

    target_splats = min(max(num_splats, len(mesh.vertices)), 80000)
    samples, face_indices = trimesh.sample.sample_surface(mesh, count=target_splats)
    normals = mesh.face_normals[face_indices]

    # High-resolution photographic texture projection from input photo
    raw_rgb = np.array(raw_img.convert("RGB")).astype(np.float32) / 255.0
    r_H, r_W, _ = raw_rgb.shape
    bx_min, by_min, _ = mesh.bounds[0]
    bx_max, by_max, _ = mesh.bounds[1]

    u_norm = np.clip((samples[:, 0] - bx_min) / max(bx_max - bx_min, 1e-6), 0.0, 1.0)
    v_norm = np.clip((by_max - samples[:, 1]) / max(by_max - by_min, 1e-6), 0.0, 1.0)

    u_px = np.clip(np.round(u_norm * (r_W - 1)).astype(int), 0, r_W - 1)
    v_px = np.clip(np.round(v_norm * (r_H - 1)).astype(int), 0, r_H - 1)
    photo_colors = raw_rgb[v_px, u_px]

    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        face_vertices = mesh.faces[face_indices]
        v_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32)
        if v_colors.max() > 1.0:
            v_colors = v_colors / 255.0
        v_col_samples = np.clip(v_colors[face_vertices].mean(axis=1), 0.0, 1.0)

        # Front-facing surfaces get crisp photo colors, back surfaces blend with inferred colors
        front_weight = np.clip((-normals[:, 2] + 0.2) / 0.5, 0.0, 1.0)[:, None]
        colors = front_weight * photo_colors + (1.0 - front_weight) * v_col_samples
    else:
        colors = photo_colors

    surface_area = mesh.area
    avg_spacing = np.sqrt(max(surface_area / target_splats, 1e-7)) * 1.5

    # Linear scales: tangential major/minor + thin normal thickness
    scales = np.zeros((target_splats, 3), dtype=np.float32)
    scales[:, 0] = avg_spacing
    scales[:, 1] = avg_spacing
    scales[:, 2] = avg_spacing * 0.25

    opacities = np.full(target_splats, 0.95, dtype=np.float32)

    # Unit quaternions (x, y, z, w) rotating [0, 0, 1] to surface normal
    rotations = np.zeros((target_splats, 4), dtype=np.float32)
    rotations[:, 0] = -normals[:, 1]
    rotations[:, 1] = normals[:, 0]
    rotations[:, 2] = 0.0
    rotations[:, 3] = 1.0 + normals[:, 2]

    antiparallel = (1.0 + normals[:, 2]) < 1e-6
    rotations[antiparallel] = [1.0, 0.0, 0.0, 0.0]

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
        cam_center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        cam_up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        cam_distance=3.5,
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
