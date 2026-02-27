from isaaclab.app import AppLauncher
import argparse

# isaac sim/kit 쪽 모듈은 AppLauncher 이후에 import되는게 안전함
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import omni.usd
from pxr import UsdPhysics  # <- 여기서도 pxr이 안 잡히면, 지금 python이 kit python이 아님

stage = omni.usd.get_context().get_stage()

root_path = "/World/envs/env_0/Robot"
root = stage.GetPrimAtPath(root_path)
print("root valid:", root.IsValid(), "path:", root_path)

count = 0
if root.IsValid():
    for prim in root.GetChildren():
        # children만이라도 찍어보자
        print("child:", prim.GetPath())
    # PrimRange는 pxr 필요. pxr 되면 아래로 확장.
    try:
        from pxr import Usd
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print("RigidBody:", prim.GetPath())
                count += 1
    except Exception as e:
        print("[WARN] PrimRange scan failed:", repr(e))

print("RigidBody count:", count)

app.close()
