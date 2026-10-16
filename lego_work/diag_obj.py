import os, bpy, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=os.path.join(HERE, "src", "Lego_856_Bulldozer.obj"),
                      forward_axis="NEGATIVE_Z", up_axis="Y")

def bbox(objs):
    pts = np.array([tuple(o.matrix_world @ v.co) for o in objs for v in o.data.vertices])
    return pts.min(0).round(2), pts.max(0).round(2), len(pts)

sus = ["LegoPlane_huge", "Plane", "Plane.001"]
for n in sus:
    o = bpy.data.objects[n]
    lo, hi, nv = bbox([o])
    print("SUS", n, "min", lo, "max", hi, "verts", nv,
          "materials", [m.name for m in o.data.materials])
rest = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in sus]
lo, hi, nv = bbox(rest)
print("REST", len(rest), "objects  min", lo, "max", hi, "verts", nv)