"""Render the office scene into a COLMAP-ready dataset with exact float depth.

    blender.exe --background "office/Office Furniture.blend" \
                --python render_office.py -- [--count 200]

Writes  nerf_synthetic/office/
    images/r_N.png            800x800 RGB (the backdrop IS the background here,
                              so no alpha compositing is needed)
    depth/r_N_depth.npz       float32 'depth' (H,W): Z-DEPTH, 0 where a ray
                              escapes the scene entirely
    transforms.json           all views in ONE file (no train/val split)
    mesh_evaluated.ply        the exact geometry that was ray-cast

Depth comes from ray/mesh intersection through each pixel CENTRE, using the
EVALUATED depsgraph geometry -- this scene has 43 Subsurf, 12 Boolean, 11
Mirror, 5 Array and 1 Geometry-Nodes modifier, so the base meshes are NOT what
Cycles renders.  Instances are walked via depsgraph.object_instances.

Conventions (identical to the airplane set):
    * transform_matrix: camera-to-world, OpenGL/Blender (x right, y up, -z view)
    * depth: z-depth along the optical axis, scene units, 0 = ray escaped
    * pixel (row r, col c) centre is (u, v) = (c + 0.5, r + 0.5)
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

W = H = 800
FOV = 0.6911112070083618
F = 0.5 * W / math.tan(0.5 * FOV)
RADIUS = 3.8
ELEV = (10.0, 60.0)
SAMPLES = 96
OUT = r"C:/Users/fatem/COLMAP/nerf_synthetic/office"


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--out", default=OUT)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return ap.parse_args(argv)


def subject_centre():
    """Centre + radius of the furniture, ignoring the big studio backdrop."""
    pts = []
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        p = np.array([o.matrix_world @ mathutils.Vector(c) for c in o.bound_box])
        if np.linalg.norm(p.max(0) - p.min(0)) > 8:          # the backdrop
            continue
        pts += [p.min(0), p.max(0)]
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)


def evaluated_bvh():
    """BVH over the RENDERED geometry: evaluated modifiers + instances."""
    dg = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    n_obj = 0
    for inst in dg.object_instances:
        ob = inst.object
        if ob.type != "MESH" or ob.original.hide_render:
            continue
        try:
            me = ob.to_mesh()
        except RuntimeError:
            continue
        if me is None or len(me.polygons) == 0:
            ob.to_mesh_clear()
            continue
        tmp = bmesh.new()
        tmp.from_mesh(me)
        tmp.transform(inst.matrix_world)
        me2 = bpy.data.meshes.new("t")
        tmp.to_mesh(me2)
        tmp.free()
        bm.from_mesh(me2)
        bpy.data.meshes.remove(me2)
        ob.to_mesh_clear()
        n_obj += 1
    bmesh.ops.triangulate(bm, faces=bm.faces)
    print("BVH: %d objects/instances, %d tris" % (n_obj, len(bm.faces)), flush=True)
    tree = BVHTree.FromBMesh(bm)
    return tree, bm


def setup(radius):
    sc = bpy.context.scene
    cam = bpy.data.objects.new("cam_render", bpy.data.cameras.new("cam_render"))
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle_x = FOV
    cam.data.clip_start, cam.data.clip_end = 0.02, 200.0
    sc.collection.objects.link(cam)
    sc.camera = cam
    r = sc.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y, r.resolution_percentage = W, H, 100
    r.film_transparent = False                   # keep the backdrop
    r.image_settings.media_type = "IMAGE"
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            if [d for d in prefs.devices if d.type == kind]:
                for d in prefs.devices:
                    d.use = d.type == kind
                sc.cycles.device = "GPU"
                print("cycles device:", kind)
                break
        except TypeError:
            continue
    return sc, cam


def poses(n, centre, radius, seed):
    """Uniform in azimuth, elevation band ELEV, looking at the subject centre."""
    rng = np.random.default_rng(seed)
    lo, hi = np.radians(ELEV)
    out = []
    for _ in range(n):
        el = np.arcsin(rng.uniform(np.sin(lo), np.sin(hi)))
        az = rng.uniform(0, 2 * np.pi)
        c = np.asarray(centre) + radius * np.array(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        fwd = np.asarray(centre) - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, -fwd, c
        out.append(M)
    return out


def depth_map(tree, M):
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
            out[i] = dist * float(d @ fwd)
    return out.reshape(H, W)


def main():
    a = args()
    centre, rad = subject_centre()
    print("subject centre", np.round(centre, 3), "radius %.3f" % rad, flush=True)
    sc, cam = setup(a.radius)
    tree, bm = evaluated_bvh()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)

    frames = []
    for n, M in enumerate(poses(a.count, centre, a.radius, a.seed)):
        t0 = time.time()
        cam.matrix_world = mathutils.Matrix(M.tolist())
        sc.render.filepath = os.path.join(a.out, "images", "r_%d.png" % n)
        bpy.ops.render.render(write_still=True)
        t1 = time.time()
        d = depth_map(tree, M)
        np.savez_compressed(os.path.join(a.out, "depth", "r_%d_depth.npz" % n), depth=d)
        frames.append({"file_path": "./images/r_%d" % n,
                       "depth_path": "./depth/r_%d_depth.npz" % n,
                       "transform_matrix": M.tolist()})
        print("r_%d  render %.1fs  depth %.1fs  hit %.1f%%  range %.2f-%.2f"
              % (n, t1 - t0, time.time() - t1, 100 * (d > 0).mean(),
                 d[d > 0].min() if (d > 0).any() else -1,
                 d.max()), flush=True)

    json.dump({"camera_angle_x": FOV, "w": W, "h": H, "fl_x": F, "fl_y": F,
               "cx": W / 2, "cy": H / 2,
               "depth_convention": "z-depth along optical axis, scene units, exact "
                                   "ray cast through pixel centre, EVALUATED geometry",
               "depth_background": 0.0,
               "pixel_centre": "row r, col c -> (u,v) = (c+0.5, r+0.5)",
               "camera_convention": "camera-to-world, OpenGL/Blender (x right, y up, -z view)",
               "capture": {"radius": a.radius, "elevation_deg": list(ELEV),
                           "azimuth": "uniform 0-360", "look_at": list(map(float, centre)),
                           "subject_radius": rad, "seed": a.seed},
               "render": {"engine": "CYCLES", "samples": SAMPLES,
                          "view_transform": "Standard", "denoising": True},
               "frames": frames},
              open(os.path.join(a.out, "transforms.json"), "w"), indent=2)

    me = bpy.data.meshes.new("evaluated")
    bm.to_mesh(me)
    ob = bpy.data.objects.new("evaluated", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.wm.ply_export(filepath=os.path.join(a.out, "mesh_evaluated.ply"),
                          export_selected_objects=True, export_uv=False,
                          export_normals=False, export_colors="NONE")
    bm.free()
    print("done ->", a.out)


if __name__ == "__main__":
    main()
