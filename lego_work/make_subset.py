import json, os, shutil, sys

SRC, OUT, N = "lego_canon", "lego30", int(sys.argv[1]) if len(sys.argv) > 1 else 30
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/depth", exist_ok=True)

meta = json.load(open(f"{SRC}/transforms.json"))
frames = meta["frames"][:N]
for n, fr in enumerate(frames, start=1):
    old = os.path.basename(fr["file_path"])              # "1", "2", ...
    shutil.copy(f"{SRC}/images/{old}.png", f"{OUT}/images/{n}.png")
    shutil.copy(f"{SRC}/depth/{old}.npz", f"{OUT}/depth/{n}.npz")
    fr["file_path"] = f"./images/{n}"
    fr["depth_path"] = f"./depth/{n}.npz"
meta["frames"] = frames
meta["views"] = f"subset of lego_canon: first {N} views (seed 0)"
json.dump(meta, open(f"{OUT}/transforms.json", "w"), indent=2)
print(N, "views ->", OUT)
