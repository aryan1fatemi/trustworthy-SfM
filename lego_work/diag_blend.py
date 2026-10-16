import os, collections, bpy

HERE = os.path.dirname(os.path.abspath(__file__))
BLEND = os.path.join(HERE, "src", "Lego_856_Bulldozer.blend")
OBJ = os.path.join(HERE, "src", "Lego_856_Bulldozer.obj")

bpy.ops.wm.open_mainfile(filepath=BLEND)
sc = bpy.context.scene
print("FRAME current", sc.frame_current, "range", sc.frame_start, sc.frame_end)

obj_names = set()
with open(OBJ) as f:
    for line in f:
        if line.startswith("o "):
            obj_names.add(line[2:].strip())
print("OBJ file objects:", len(obj_names))

dg = bpy.context.evaluated_depsgraph_get()
meshes = [o for o in sc.objects if o.type == "MESH"]
print("MESH objects in scene:", len(meshes))
print("OTHER types:", dict(collections.Counter(o.type for o in sc.objects if o.type != "MESH")))
print("MODIFIER types:", dict(collections.Counter(m.type for o in meshes for m in o.modifiers)))

render, hidden = [], []
for o in meshes:
    (hidden if (o.hide_render or not o.visible_get()) else render).append(o)
print("HIDDEN:", [o.name for o in hidden])

raw = sum(len(o.data.vertices) for o in render)
ev_total = 0
for o in render:
    ev = o.evaluated_get(dg)
    me = ev.to_mesh()
    ev_total += len(me.vertices)
    ev.to_mesh_clear()
print("RENDERABLE", len(render), "| raw verts", raw, "| evaluated verts", ev_total,
      "| OBJ for comparison: about 1,090,045 (incl. the 3 dropped parts)")

names = {o.name for o in render}
print("IN BLEND NOT IN OBJ:", len(names - obj_names), sorted(names - obj_names)[:15])
print("IN OBJ NOT IN BLEND:", len(obj_names - names), sorted(obj_names - names)[:15])
