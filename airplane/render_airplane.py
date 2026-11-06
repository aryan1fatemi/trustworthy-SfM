"""Render the airplane dataset from the .obj, with exact float depth, images named 1..N.

Run with Blender (headless), from any folder:
    "C:/Program Files/Blender Foundation/Blender 5.2/blender.exe" --background --factory-startup ^
        --python render_airplane.py -- [--count 200] [--out renders]

Writes  nerf_synthetic/airplane/<out>/   (default: renders/)
    images/1.png ... images/N.png     RGB on a WHITE background (ready for COLMAP; no alpha)
    depth/1.npz ... depth/N.npz       float32 'depth' (H,W): Z-DEPTH, 0 = background
    transforms.json                   one file, NeRF format + keys documenting conventions
    mesh_normalized.obj               geometry in the SAME frame as the poses

Viewpoints: identical to the earlier train/val renders when --count 200 and --seed 0:
    images 1..100   = train r_0..r_99   (seed)
    images 101..200 = val   r_0..r_99   (seed + 1000)
so results stay comparable with the existing reconstruction.

Depth is not taken from a render pass. It is the exact ray/mesh intersection through each pixel
CENTRE (BVH ray cast), converted to z-depth.

Conventions:
    * transform_matrix is camera-to-world, Blender/OpenGL camera (x right, y up, looks along -z),
      as in NeRF-synthetic. Flip camera y, z for OpenCV / COLMAP.
    * depth is z-depth along the optical axis, scene units, 0 where the centre ray misses the mesh.
    * pixel (row r, col c) is centred at image coords (u, v) = (c + 0.5, r + 0.5), the COLMAP
      convention; principal point (400, 400).
    * the white background is composited in the saved sRGB values: rgb' = a * rgb + (1 - a),
      the same as PIL alpha compositing in the earlier prepare_images.py.
"""
import os
import sys
import json
import math
import time
import argparse
import numpy as np
import bpy
import bmesh
import mathutils
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))                   # nerf_synthetic/airplane/
SRC = os.path.join(HERE, "..", "Airplane_v1", "11803_Airplane_v1_l1.obj")

W = H = 800
ANGLE_X = 0.6911112070083618          # same field of view as chair
F = 0.5 * W / math.tan(0.5 * ANGLE_X)
RADIUS = 4.0311                        # same camera distance as chair
OBJ_RADIUS = 1.30                      # object bounding-sphere radius after normalisation
SAMPLES = 64


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200, help="number of views")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="renders", help="output folder, relative to the airplane folder")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bpy.ops.wm.obj_import(filepath=os.path.abspath(SRC), forward_axis="Y", up_axis="Z",
                          global_scale=1.0)
    meshes = [o for o in sc.objects if o.type == "MESH"]

    # normalise: centre the bounding box at the origin, fit a sphere of OBJ_RADIUS
    pts = np.array([o.matrix_world @ v.co for o in meshes for v in o.data.vertices])
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    scale = OBJ_RADIUS / np.linalg.norm(pts - centre, axis=1).max()
    T = (mathutils.Matrix.Scale(scale, 4) @
         mathutils.Matrix.Translation(mathutils.Vector(-centre)))
    for o in meshes:
        o.matrix_world = T @ o.matrix_world

    # lighting: soft white environment + one sun, so textures read clearly
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = 0.8
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    sc.collection.objects.link(sun)

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle_x = ANGLE_X
    cam.data.clip_start, cam.data.clip_end = 0.05, 100.0
    sc.collection.objects.link(cam)
    sc.camera = cam

    r = sc.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.film_transparent = True                          # alpha = object, composited below
    r.image_settings.media_type = "IMAGE"
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "Standard"      # plain sRGB, no AgX curve
    sc.view_settings.look = "None"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    _pick_device(sc)

    info = {"source_obj": os.path.abspath(SRC), "import_axes": "forward=Y up=Z",
            "normalisation": {"centre_before_scale": centre.tolist(),
                              "scale": float(scale),
                              "note": "X_scene = scale * (X_obj_imported - centre)"}}
    return sc, cam, meshes, info


def _pick_device(sc):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                sc.cycles.device = "GPU"
                print("cycles device:", kind, [d.name for d in devs])
                return
        except TypeError:
            continue
    sc.cycles.device = "CPU"
    print("cycles device: CPU")


def world_bvh(meshes):
    """One BVH over every mesh, in world (= scene) coordinates."""
    bm = bmesh.new()
    for o in meshes:
        tmp = bmesh.new()
        tmp.from_mesh(o.data)
        tmp.transform(o.matrix_world)
        me = bpy.data.meshes.new("tmp")
        tmp.to_mesh(me)
        tmp.free()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree


# --------------------------------------------------------------------------
def poses(n, seed):
    """Uniform on the upper hemisphere, looking at the origin, world-Z up."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z = rng.uniform(0.05, 1.0)                  # avoid exactly grazing views
        phi = rng.uniform(0, 2 * np.pi)
        c = RADIUS * np.array([np.sqrt(1 - z * z) * np.cos(phi),
                               np.sqrt(1 - z * z) * np.sin(phi), z])
        fwd = -c / np.linalg.norm(c)                 # camera looks along -z_cam
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, -fwd, c
        out.append(M)
    return out


def all_poses(count, seed):
    """first half from `seed`, second half from `seed + 1000` (= the earlier train / val)."""
    n1 = (count + 1) // 2
    return poses(n1, seed) + poses(count - n1, seed + 1000)


def depth_map(tree, M):
    """Z-depth at every pixel centre by exact ray casting. 0 = miss."""
    R, c = M[:3, :3], M[:3, 3]
    rr, cc = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(cc + 0.5 - W / 2) / F, -(rr + 0.5 - H / 2) / F,
                      -np.ones_like(rr, float)], -1).reshape(-1, 3)
    d_w = d_cam @ R.T
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    fwd = -R[:, 2]
    origin = mathutils.Vector(c)
    out = np.zeros(H * W, np.float32)
    rc = tree.ray_cast
    for i, d in enumerate(d_w):
        loc, _, _, dist = rc(origin, mathutils.Vector(d))
        if loc is not None:
            out[i] = dist * float(d @ fwd)           # ray distance -> z-depth
    return out.reshape(H, W)


def composite_on_white(rgba_path, out_path):
    """rgb' = a * rgb + (1 - a) on the stored 8-bit sRGB values; saves RGB PNG."""
    src = bpy.data.images.load(rgba_path)
    px = np.array(src.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(src)
    a = px[..., 3:4]
    rgb = px[..., :3] * a + (1.0 - a)
    rgb = np.round(rgb * 255) / 255                  # keep exact 8-bit levels
    dst = bpy.data.images.new("out", W, H, alpha=False)
    dst.pixels[:] = np.concatenate([rgb, np.ones_like(a)], -1).ravel()
    dst.filepath_raw = out_path
    dst.file_format = "PNG"
    dst.save()                                       # raw buffer, no colour management
    bpy.data.images.remove(dst)


# --------------------------------------------------------------------------
def main():
    a = args()
    OUT = os.path.join(HERE, a.out)
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "depth"), exist_ok=True)
    tmp = os.path.join(OUT, "_rgba_tmp.png")
    sc, cam, meshes, info = build_scene()
    tree = world_bvh(meshes)

    frames = []
    n1 = (a.count + 1) // 2
    for n, M in enumerate(all_poses(a.count, a.seed), start=1):
        t0 = time.time()
        cam.matrix_world = mathutils.Matrix(M.tolist())
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        composite_on_white(tmp, os.path.join(OUT, "images", "%d.png" % n))
        t1 = time.time()
        depth = depth_map(tree, M)
        np.savez_compressed(os.path.join(OUT, "depth", "%d.npz" % n), depth=depth)
        origin = ("train r_%d" % (n - 1)) if n <= n1 else ("val r_%d" % (n - 1 - n1))
        frames.append({"file_path": "./images/%d" % n, "depth_path": "./depth/%d.npz" % n,
                       "same_view_as": origin, "transform_matrix": M.tolist()})
        print("%3d.png (%s)  render %.1fs  depth %.1fs  object px %d"
              % (n, origin, t1 - t0, time.time() - t1, int((depth > 0).sum())), flush=True)
    if os.path.exists(tmp):
        os.remove(tmp)

    json.dump({"camera_angle_x": ANGLE_X, "frames": frames,
               "w": W, "h": H, "fl_x": F, "fl_y": F, "cx": W / 2, "cy": H / 2,
               "background": "white, composited (rgb' = a*rgb + 1 - a)",
               "depth_convention": "z-depth along optical axis, scene units, "
                                   "exact ray cast through pixel centre",
               "depth_background": 0.0,
               "pixel_centre": "row r, col c -> (u,v) = (c+0.5, r+0.5)",
               "camera_convention": "camera-to-world, OpenGL/Blender (x right, y up, -z view)",
               "views": "1..%d from seed %d, %d..%d from seed %d" % (n1, a.seed, n1 + 1, a.count, a.seed + 1000),
               "render": {"engine": "CYCLES", "samples": SAMPLES, "view_transform": "Standard"},
               **info},
              open(os.path.join(OUT, "transforms.json"), "w"), indent=2)

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.ops.wm.obj_export(filepath=os.path.join(OUT, "mesh_normalized.obj"),
                          export_selected_objects=True, export_materials=False,
                          export_uv=False, export_normals=False,
                          forward_axis="Y", up_axis="Z")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
