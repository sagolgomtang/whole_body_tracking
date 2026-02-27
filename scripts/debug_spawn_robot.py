import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# AppLauncher args (여기에 --headless 포함)
AppLauncher.add_app_launcher_args(parser)

# task 지정
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
args = parser.parse_args()

# 먼저 Isaac Sim app 띄우기
app = AppLauncher(args).app

# 그 다음 imports
import gymnasium as gym
import omni.usd
from pxr import Usd, UsdPhysics

# --- add this before gym.make ---
import whole_body_tracking.tasks  # 또는 import whole_body_tracking.tasks.tracking
from gymnasium.envs.registration import registry
print("Registered Tracking envs:", [k for k in registry.keys() if "Tracking" in k])
# --------------------------------


# env 생성 (이 시점에 /World/envs/... 생성됨)
# env 생성 전에 spec에서 cfg class 꺼내서 cfg 만들어서 넣어야 함
spec = gym.spec(args.task)
env_cfg_cls = spec.kwargs["env_cfg_entry_point"]   # 예: G1FlatEnvCfg class
env_cfg = env_cfg_cls()                            # 인스턴스 생성

env = gym.make(args.task, cfg=env_cfg, render_mode=None)
env.reset()

# 한 번 reset해서 scene 구성 완료시키기
env.reset()

stage = omni.usd.get_context().get_stage()

# env_0 아래를 통째로 스캔해서 Robot이 어디 생겼는지 찾기
root_env = stage.GetPrimAtPath("/World/envs/env_0")
print("env_0 valid:", root_env.IsValid())

robot_candidates = []
if root_env.IsValid():
    for prim in Usd.PrimRange(root_env):
        p = str(prim.GetPath())
        # 로봇 후보: 이름에 Robot 포함
        if "Robot" in p:
            robot_candidates.append(p)

print("Robot candidates (paths containing 'Robot'):")
for p in robot_candidates[:50]:
    print("  ", p)

# RigidBody가 실제로 어디에 있는지도 같이 찍기
rb_paths = []
if root_env.IsValid():
    for prim in Usd.PrimRange(root_env):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_paths.append(str(prim.GetPath()))

print(f"RigidBody count under /World/envs/env_0: {len(rb_paths)}")
for p in rb_paths[:50]:
    print("  RigidBody:", p)

# 종료
env.close()
app.close()
