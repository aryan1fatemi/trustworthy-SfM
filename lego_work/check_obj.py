# import bpy
# bpy.ops.wm.read_factory_settings(use_empty=True)
# bpy.ops.wm.obj_import(filepath="src/Lego_856_Bulldozer.obj", forward_axis="Y", up_axis="Z")
# for o in bpy.data.objects:
#     print("OBJ", o.name, o.type, "dims", tuple(round(x, 3) for x in o.dimensions))

import os
import bpy
HERE = os.path.dirname(os.path.abspath(__file__))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=os.path.join(HERE, "src", "Lego_856_Bulldozer.obj"), forward_axis="Y", up_axis="Z")
for o in bpy.data.objects:
    print("OBJ", o.name, o.type, "dims", tuple(round(x, 3) for x in o.dimensions))
