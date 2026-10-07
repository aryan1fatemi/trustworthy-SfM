"""Render a NeRF-synthetic-style dataset from Airplane_v1 with exact float depth.

Run with Blender (headless):
    blender.exe --background --factory-startup --python render_airplane.py -- [--count N]

Writes  nerf_synthetic/airplane/
    train/r_N.png, val/r_N.png           RGBA, like NeRF-synthetic (alpha = object)
    train/r_N_depth.npz, val/...         float32 'depth' (H,W): Z-DEPTH, 0 = background
    transforms_train.json / _val.json    NeRF format + keys documenting conventions
    mesh_normalized.obj                  geometry in the SAME frame as the poses

Depth is not taken from a render pass.  It is the exact ray/mesh intersection
through each pixel CENTRE (BVH ray cast), converted to z-depth.  So its
definition is fixed in code: no anti-aliasing mixing, no pass-format ambiguity.

Conventions:
    * transform_matrix is camera-to-world, Blender/OpenGL camera (x right, y up,
      looks along -z), exactly as in NeRF-synthetic.  Flip camera y,z for OpenCV.
    * depth is z-depth along the optical axis, scene units, 0 where the centre
      ray misses the mesh.
    * pixel (row r, col c) is centred at image coords (u, v) = (c + 0.5, r + 0.5),
      the COLMAP convention; principal point (400, 400).
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))       # nerf_synthetic/
SRC = os.path.join(ROOT, "Airplane_v1", "11803_Airplane_v1_l1.obj")
OUT = os.path.join(ROOT, "airplane")

W = H = 800
ANGLE_X = 0.6911112070083618          # same field of view as chair
F = 0.5 * W / math.tan(0.5 * ANGLE_X)
RADIUS = 4.0311                        # same camera distance as chair
OBJ_RADIUS = 1.30                      # object bounding-sphere radius after normalisation
SAMPLES = 64


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100, help="views per split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="train,val")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
def build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bpy.ops.wm.obj_import(filepath=SRC, forward_axis="Y", up_axis="Z",
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
    r.film_transparent = True
    r.image_settings.media_type = "IMAGE"
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    r.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "Standard"      # plain sRGB, no AgX curve
    sc.view_settings.look = "None"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    _pick_device(sc)

    info = {"source_obj": SRC, "import_axes": "forward=Y up=Z",
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


# --------------------------------------------------------------------------
def main():
    a = args()
    sc, cam, meshes, info = build_scene()
    tree = world_bvh(meshes)
    os.makedirs(OUT, exist_ok=True)

    for s_i, split in enumerate(a.splits.split(",")):
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
        frames = []
        for n, M in enumerate(poses(a.count, a.seed + 1000 * s_i)):
            t0 = time.time()
            stem = os.path.join(OUT, split, "r_%d" % n)
            cam.matrix_world = mathutils.Matrix(M.tolist())
            sc.render.filepath = stem + ".png"
            bpy.ops.render.render(write_still=True)
            t1 = time.time()
            depth = depth_map(tree, M)
            np.savez_compressed(stem + "_depth.npz", depth=depth)
            frames.append({"file_path": "./%s/r_%d" % (split, n),
                           "depth_path": "./%s/r_%d_depth.npz" % (split, n),
                           "transform_matrix": M.tolist()})
            print("%s r_%d  render %.1fs  depth %.1fs  object px %d"
                  % (split, n, t1 - t0, time.time() - t1, int((depth > 0).sum())),
                  flush=True)

        json.dump({"camera_angle_x": ANGLE_X, "frames": frames,
                   "w": W, "h": H, "fl_x": F, "fl_y": F, "cx": W / 2, "cy": H / 2,
                   "depth_convention": "z-depth along optical axis, scene units, "
                                       "exact ray cast through pixel centre",
                   "depth_background": 0.0,
                   "pixel_centre": "row r, col c -> (u,v) = (c+0.5, r+0.5)",
                   "camera_convention": "camera-to-world, OpenGL/Blender "
                                        "(x right, y up, -z view)",
                   "render": {"engine": "CYCLES", "samples": SAMPLES,
                              "view_transform": "Standard"},
                   **info},
                  open(os.path.join(OUT, "transforms_%s.json" % split), "w"), indent=2)

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
