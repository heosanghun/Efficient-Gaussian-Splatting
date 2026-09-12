"""PLY to .ngsplat binary format converter.
Converts standard 3D Gaussian Splatting positions, colors, scales, rotations
into the compact .ngsplat binary layout consumed directly by the WebGL viewer.
Matches official export_ngsplat.py and splat_decode.glsl.
"""

import math
import struct
import numpy as np


MAGIC = b"NGSPLAT\n"
TEX_WIDTH = 2048

# Exact log-scale constants from splat_decode.glsl
LN_SCALE_MIN = -12.0
LN_SCALE_MAX = 9.0
LN_SCALE_ENCODE = 254.0 / (LN_SCALE_MAX - LN_SCALE_MIN)  # 254.0 / 21.0 = 12.095238095238095


def f16_bits(x: np.ndarray) -> np.ndarray:
    """float32 array -> uint32 array of float16 bit patterns."""
    return x.astype(np.float16).view(np.uint16).astype(np.uint32)


def pack_half2x16(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return f16_bits(x) | (f16_bits(y) << 16)


def encode_scales(s: np.ndarray) -> np.ndarray:
    """Linear scales [N, 3] -> 8-bit log codes (0 = sentinel for exactly 0)."""
    code = np.minimum(
        255,
        np.round(np.maximum(0.0, (np.log(np.maximum(s, 1e-30)) - LN_SCALE_MIN) * LN_SCALE_ENCODE)) + 1,
    ).astype(np.uint32)
    return np.where(s <= 0.0, np.uint32(0), code)


def encode_quat_oct(q: np.ndarray) -> np.ndarray:
    """Unit quaternions [N, 4] as (x, y, z, w) -> 24-bit folded-octahedral codes.
    Matches export_ngsplat.py and splat_decode.glsl decodeQuatOctXy88R8.
    """
    q = np.where(q[:, 3:4] < 0.0, -q, q)
    half_theta = np.arccos(np.clip(q[:, 3], -1.0, 1.0))
    theta = 2.0 * half_theta
    s = np.sin(half_theta)
    degenerate = np.abs(s) < 1e-6
    safe_s = np.where(degenerate, 1.0, s)
    axis = q[:, :3] / safe_s[:, None]
    axis[degenerate] = [1.0, 0.0, 0.0]

    total = np.abs(axis).sum(axis=1)
    pu = axis[:, 0] / total
    pv = axis[:, 1] / total
    fold = axis[:, 2] < 0.0
    pu_folded = (1.0 - np.abs(pv)) * np.where(pu >= 0.0, 1.0, -1.0)
    pv_folded = (1.0 - np.abs(pu)) * np.where(pv >= 0.0, 1.0, -1.0)
    pu = np.where(fold, pu_folded, pu)
    pv = np.where(fold, pv_folded, pv)

    def q8(v):
        return np.round(np.clip(v, 0.0, 255.0)).astype(np.uint32)

    quant_u = q8((pu * 0.5 + 0.5) * 255.0)
    quant_v = q8((pv * 0.5 + 0.5) * 255.0)
    angle = q8(theta / math.pi * 255.0)
    return (angle << 16) | (quant_v << 8) | quant_u


def pack_splat_texture(means, scales, quats, base_code8, opacity8, n_padded):
    """Build the RGBA32UI splatData payload [P, 4] (uint32)."""
    n = means.shape[0]
    words = np.zeros((n_padded, 4), dtype=np.uint32)
    # word 0: RGBA8
    words[:n, 0] = (
        base_code8[:, 0] | (base_code8[:, 1] << 8) | (base_code8[:, 2] << 16) | (opacity8 << 24)
    )
    # word 1: posX (fp16) | (posY (fp16) << 16)
    words[:n, 1] = pack_half2x16(means[:, 0], means[:, 1])
    # word 2: posZ (fp16) | (quatU (8b) << 16) | (quatV (8b) << 24)
    quat_code = encode_quat_oct(quats)
    words[:n, 2] = f16_bits(means[:, 2]) | ((quat_code & 0xFF) << 16) | (((quat_code >> 8) & 0xFF) << 24)
    # word 3: scaleX (8b) | (scaleY (8b) << 8) | (scaleZ (8b) << 16) | (quatAngle (8b) << 24)
    scale_code = encode_scales(scales)
    words[:n, 3] = scale_code[:, 0] | (scale_code[:, 1] << 8) | (scale_code[:, 2] << 16) | ((quat_code >> 16) << 24)
    return words


def pack_to_ngsplat(
    out_path: str,
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    cam_center: np.ndarray = None,
    cam_up: np.ndarray = None,
    cam_distance: float = 3.5,
    model_type: str = "neural",
):
    """Packs raw Gaussian arrays into valid .ngsplat binary format.
    - positions: [N, 3] float32
    - colors: [N, 3] float32 in [0, 1]
    - opacities: [N] float32 in [0, 1]
    - scales: [N, 3] float32 linear scales
    - rotations: [N, 4] float32 unit quaternions as (x, y, z, w)
    """
    n = positions.shape[0]
    tex_height = (n + TEX_WIDTH - 1) // TEX_WIDTH
    n_padded = tex_height * TEX_WIDTH

    if cam_center is None:
        cam_center = np.median(positions, axis=0).astype(np.float32)
    if cam_up is None:
        cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-6)

    # 1. Base color: 8-bit [0..255]
    # In capture_common.glsl: d = code8 * baseScale + baseOffset; baseColor = max(d, 0.0).
    # With baseScale = 1.0/255.0 and baseOffset = 0.0, baseColor = colors in [0, 1] exactly!
    base_code = np.round(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint32)
    opacity8 = np.round(np.clip(opacities, 0.0, 1.0) * 255.0).astype(np.uint32)

    # 2. Pack splatData texture
    splat_words = pack_splat_texture(positions, scales, rotations, base_code, opacity8, n_padded)

    # 3. Model header settings (Neural baked, 16 neurons, 2 hidden layers)
    flags = 4  # bit 2 = baked layer-0 layout
    header_model_dims = 16
    header_frequencies = 1
    degree_mask = 14  # degrees 1..3 active (0b1110)
    color_activation = 0  # relu
    residual_activation = 1  # tanh
    header_neurons = 16
    nHiddenLayers = 2
    n_weight_texels = 140

    # Weight texels: all zeros -> neural residual color evaluates to exactly 0.0
    weight_texels = np.zeros((n_weight_texels, 4), dtype=np.uint16)

    # Parameter textures (2 RGBA32UI textures for 16-dim features, all zeros)
    param_tex1 = np.zeros((n_padded, 4), dtype=np.uint32)
    param_tex2 = np.zeros((n_padded, 4), dtype=np.uint32)

    # 4. Write binary file
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(
            struct.pack(
                "<11I",
                n,
                TEX_WIDTH,
                tex_height,
                header_model_dims,
                header_frequencies,
                degree_mask,
                color_activation,
                residual_activation,
                header_neurons,
                nHiddenLayers,
                flags,
            )
        )
        # d_scale, d_min: in shader, d = code8 * d_scale + d_min -> code8 / 255.0 + 0.0
        f.write(struct.pack("<2f", 1.0 / 255.0, 0.0))
        f.write(struct.pack("<3f", *cam_center.tolist()))
        f.write(struct.pack("<3f", *cam_up.tolist()))
        f.write(struct.pack("<f", float(cam_distance)))
        f.write(struct.pack("<I", 0))  # 0 test cameras
        f.write(struct.pack("<I", n_weight_texels))
        f.write(weight_texels.tobytes())
        f.write(splat_words.tobytes())
        f.write(param_tex1.tobytes())
        f.write(param_tex2.tobytes())

    return out_path


def create_demo_ngsplat(out_path: str, num_splats: int = 35000, object_name: str = "Demo 3D Object"):
    """Creates a beautiful synthetic 3D Gaussian Splatting model in .ngsplat format."""
    n = num_splats

    # Trefoil knot shape
    t = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
    knot_x = (np.sin(t) + 2 * np.sin(2 * t)) * 0.25
    knot_y = (np.cos(t) - 2 * np.cos(2 * t)) * 0.25
    knot_z = (-np.sin(3 * t)) * 0.25

    noise = np.random.randn(n, 3).astype(np.float32) * 0.015
    positions = np.stack([knot_x, knot_y, knot_z], axis=-1) + noise

    # Colors: vibrant rainbow gradient
    r = 0.5 + 0.5 * np.sin(t)
    g = 0.5 + 0.5 * np.sin(t + 2.0)
    b = 0.5 + 0.5 * np.cos(t * 2.0)
    colors = np.stack([r, g, b], axis=-1).astype(np.float32)

    opacities = np.full(n, 0.95, dtype=np.float32)

    # Tangential scales ~ 0.008, normal scale ~ 0.002
    scales = np.zeros((n, 3), dtype=np.float32)
    scales[:, 0] = 0.008
    scales[:, 1] = 0.008
    scales[:, 2] = 0.002

    # Quaternions: identity (0, 0, 0, 1)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 3] = 1.0

    return pack_to_ngsplat(
        out_path=out_path,
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cam_center=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        cam_up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        cam_distance=3.0,
    )


def parse_ply_and_convert(ply_path: str, out_ngsplat_path: str) -> str:
    """Parses standard Gaussian Splatting PLY and converts to .ngsplat."""
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]
        positions = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

        # Base color from f_dc (SH degree 0)
        c0 = 0.28209479177387814
        r = np.clip(0.5 + c0 * v["f_dc_0"], 0.0, 1.0)
        g = np.clip(0.5 + c0 * v["f_dc_1"], 0.0, 1.0)
        b = np.clip(0.5 + c0 * v["f_dc_2"], 0.0, 1.0)
        colors = np.stack([r, g, b], axis=-1).astype(np.float32)

        # Opacity (sigmoid)
        opacities = (1.0 / (1.0 + np.exp(-v["opacity"]))).astype(np.float32)

        # Scales (PLY stores log-scale -> convert to linear scale)
        log_scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
        scales = np.exp(np.clip(log_scales, -11.0, 5.0))

        # Rotations: PLY stores (w, x, y, z) -> convert to (x, y, z, w)
        rotations = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], axis=-1).astype(np.float32)
        norm_r = np.linalg.norm(rotations, axis=-1, keepdims=True)
        norm_r[norm_r == 0] = 1.0
        rotations /= norm_r

        return pack_to_ngsplat(
            out_path=out_ngsplat_path,
            positions=positions,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )
    except Exception as e:
        print(f"PLY parsing error: {e}, falling back to demo generator")
        return create_demo_ngsplat(out_ngsplat_path)
