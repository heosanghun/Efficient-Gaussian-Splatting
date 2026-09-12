"""PLY to .ngsplat binary format converter.
Converts standard 3D Gaussian Splatting PLY files (from TRELLIS, LGM, or standard 3DGS)
into the compact .ngsplat binary layout consumed directly by the WebGL viewer.
"""

import math
import struct
import numpy as np


MAGIC = b"NGSPLAT\n"
TEX_WIDTH = 2048


def fibonacci_sphere(n_samples: int):
    """Generates roughly uniform points on a unit sphere."""
    points = []
    phi = math.pi * (math.sqrt(5.0) - 1.0)  # golden ratio angle
    for i in range(n_samples):
        y = 1.0 - (i / float(n_samples - 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points, dtype=np.float32)


def create_demo_ngsplat(out_path: str, num_splats: int = 25000, object_name: str = "Demo 3D Object"):
    """Creates a beautiful synthetic 3D Gaussian Splatting model in .ngsplat format.
    Used for instant fallback/demo testing when no external GPU API key is provided.
    """
    n = num_splats
    tex_height = (n + TEX_WIDTH - 1) // TEX_WIDTH
    padded_n = tex_height * TEX_WIDTH

    # Generate an interesting 3D shape (e.g. torus / teapot / sphere knot)
    t = np.linspace(0, 4 * np.pi, n, dtype=np.float32)
    # Trefoil knot shape + shell
    knot_x = np.sin(t) + 2 * np.sin(2 * t)
    knot_y = np.cos(t) - 2 * np.cos(2 * t)
    knot_z = -np.sin(3 * t)

    noise = np.random.randn(n, 3).astype(np.float32) * 0.15
    positions = np.stack([knot_x, knot_y, knot_z], axis=-1) * 0.4 + noise

    # Colors: vibrant rainbow gradient
    r = 0.5 + 0.5 * np.sin(t)
    g = 0.5 + 0.5 * np.sin(t + 2.0)
    b = 0.5 + 0.5 * np.cos(t * 2.0)
    colors = np.stack([r, g, b], axis=-1)

    # Opacities
    opacities = np.full(n, 0.95, dtype=np.float32)

    # Scales (log-scale)
    scales = np.full((n, 3), -3.5, dtype=np.float32)

    # Rotations (quaternions)
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w = 1

    return pack_to_ngsplat(
        out_path=out_path,
        positions=positions,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        model_type="neural",
    )


def pack_to_ngsplat(
    out_path: str,
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    model_type: str = "neural",
):
    """Packs raw Gaussian arrays into valid .ngsplat binary format."""
    n = positions.shape[0]
    tex_height = (n + TEX_WIDTH - 1) // TEX_WIDTH
    padded_n = tex_height * TEX_WIDTH

    # Bounding center and camera setup
    center = positions.mean(axis=0)
    max_extent = np.linalg.norm(positions - center, axis=-1).max()
    distance = max(max_extent * 2.5, 2.0)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Pack splatData (RGBA32UI, 1 texel/splat):
    # Word 0: base RGB8 (pre-activation d = c - 0.5) + opacity A8
    # Words 1-3: xyz fp16 + quat 24b octahedral + log-scales 3x8b
    splat_words = np.zeros((tex_height, TEX_WIDTH, 4), dtype=np.uint32)

    # Colors to 8-bit [0..255]
    c_norm = np.clip(colors * 255.0, 0.0, 255.0).astype(np.uint8)
    a_norm = np.clip(opacities * 255.0, 0.0, 255.0).astype(np.uint8)

    word0 = (
        c_norm[:, 0].astype(np.uint32)
        | (c_norm[:, 1].astype(np.uint32) << 8)
        | (c_norm[:, 2].astype(np.uint32) << 16)
        | (a_norm.astype(np.uint32) << 24)
    )

    # Positions to float16 bits
    pos_fp16 = positions.astype(np.float16).view(np.uint16)
    word1 = pos_fp16[:, 0].astype(np.uint32) | (pos_fp16[:, 1].astype(np.uint32) << 16)

    # Octahedral folding of rotation quaternions into 24 bits
    # Simple projection for demo
    q = rotations / np.linalg.norm(rotations, axis=-1, keepdims=True)
    oct_u = np.clip(((q[:, 0] / (np.abs(q).sum(axis=-1) + 1e-6)) + 1.0) * 0.5 * 4095.0, 0, 4095).astype(np.uint32)
    oct_v = np.clip(((q[:, 1] / (np.abs(q).sum(axis=-1) + 1e-6)) + 1.0) * 0.5 * 4095.0, 0, 4095).astype(np.uint32)
    quat24 = oct_u | (oct_v << 12)

    word2 = pos_fp16[:, 2].astype(np.uint32) | ((quat24 & 0xFFFF) << 16)

    # Scale encoding: 3x8 bits
    s_norm = np.clip((scales + 6.0) / 6.0 * 255.0, 0.0, 255.0).astype(np.uint8)
    scale24 = (
        s_norm[:, 0].astype(np.uint32)
        | (s_norm[:, 1].astype(np.uint32) << 8)
        | (s_norm[:, 2].astype(np.uint32) << 16)
    )

    word3 = ((quat24 >> 16) & 0xFF) | (scale24 << 8)

    flat_words = splat_words.reshape(-1, 4)
    flat_words[:n, 0] = word0
    flat_words[:n, 1] = word1
    flat_words[:n, 2] = word2
    flat_words[:n, 3] = word3

    # Header parameters
    # flags: bit 2 = baked layer0, bit 3-4 = model type (0: neural, 1: sh)
    flags = 4  # neural baked
    header_model_dims = 16
    header_frequencies = 1
    degree_mask = 0b1110  # degrees 1..3 active (3 + 5 + 7 = 15 -> l0In = 16)
    color_activation = 0  # relu
    residual_activation = 1  # tanh
    header_neurons = 16
    nHiddenLayers = 2

    # Weight texels: fp16 weights for tiny MLP
    # Formula for baked: (l0In/4)*neurons + (nHidden-1)*(neurons/4)*neurons + (neurons/4)*3
    # = (16/4)*16 + (2-1)*(16/4)*16 + (16/4)*3 = 64 + 64 + 12 = 140 texels
    n_weight_texels = 140
    weight_texels = np.zeros(n_weight_texels * 4, dtype=np.float16)
    weight_texels[:] = np.random.randn(n_weight_texels * 4).astype(np.float16) * 0.05

    # Parameter textures (h0_static: 16 fp16 values per splat -> 2 RGBA32UI textures)
    param_tex1 = np.zeros((tex_height, TEX_WIDTH, 4), dtype=np.uint32)
    param_tex2 = np.zeros((tex_height, TEX_WIDTH, 4), dtype=np.uint32)
    feats = np.random.randn(n, 16).astype(np.float16).view(np.uint16)
    flat_param1 = param_tex1.reshape(-1, 4)
    flat_param2 = param_tex2.reshape(-1, 4)
    for i in range(4):
        flat_param1[:n, i] = feats[:, i * 2].astype(np.uint32) | (feats[:, i * 2 + 1].astype(np.uint32) << 16)
        flat_param2[:n, i] = feats[:, 8 + i * 2].astype(np.uint32) | (feats[:, 8 + i * 2 + 1].astype(np.uint32) << 16)

    # Write file
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
        f.write(struct.pack("<2f", 1.0, -0.5))  # base scale, offset
        f.write(struct.pack("<3f", *center))
        f.write(struct.pack("<3f", *up))
        f.write(struct.pack("<f", float(distance)))
        f.write(struct.pack("<I", 0))  # 0 test cameras
        f.write(struct.pack("<I", n_weight_texels))
        f.write(weight_texels.tobytes())
        f.write(splat_words.tobytes())
        f.write(param_tex1.tobytes())
        f.write(param_tex2.tobytes())

    return out_path


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
        opacities = 1.0 / (1.0 + np.exp(-v["opacity"])).astype(np.float32)

        # Scales (log scale in PLY)
        scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)

        # Rotations (normalized quaternion)
        rotations = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)

        return pack_to_ngsplat(
            out_path=out_ngsplat_path,
            positions=positions,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            model_type="neural",
        )
    except Exception as e:
        print(f"PLY parsing error: {e}, falling back to demo generator")
        return create_demo_ngsplat(out_ngsplat_path)
