from pathlib import Path
import torch
import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path("/home/user/rl_ws/src/whole_body_tracking/logs/rsl_rl/g1_flat")

MEAN_NAME = "actor_obs_norm_mean"
STD_NAME  = "actor_obs_norm_std"
SUB_NAME  = "Normalize_Sub"
DIV_NAME  = "Normalize_Div"
OBS_NORM  = "obs_norm"
OBS_CTR   = "obs_centered"

def patch_policy_onnx(run_dir: Path):
    ckpt = run_dir / "model_best.pt"
    onnx_path = run_dir / "exported" / "policy.onnx"

    if not ckpt.is_file():
        return False, "no model_best.pt"
    if not onnx_path.is_file():
        return False, "no exported/policy.onnx"

    d = torch.load(str(ckpt), map_location="cpu")
    sd = d.get("model_state_dict", None)
    if sd is None:
        return False, "no model_state_dict"

    if "actor_obs_normalizer._mean" not in sd or "actor_obs_normalizer._std" not in sd:
        return False, "no actor_obs_normalizer mean/std in ckpt"

    mean = sd["actor_obs_normalizer._mean"].detach().cpu().numpy().astype(np.float32).reshape(-1)
    std  = sd["actor_obs_normalizer._std" ].detach().cpu().numpy().astype(np.float32).reshape(-1)
    obs_dim = int(mean.size)

    m = onnx.load(str(onnx_path))
    g = m.graph

    # controller 요구사항: input 'obs', output 'actions'
    inputs = [i.name for i in g.input]
    outputs = [o.name for o in g.output]
    if "obs" not in inputs:
        return False, f"input 'obs' not found (inputs={inputs})"
    if "actions" not in outputs:
        return False, f"output 'actions' not found (outputs={outputs})"

    # obs input shape 확인(있으면)
    for inp in g.input:
        if inp.name == "obs":
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            if len(dims) == 2 and dims[1] not in (0, obs_dim):  # 0은 dynamic/unknown
                return False, f"obs_dim mismatch: ckpt={obs_dim}, onnx={dims[1]}"

    # --- 기존 패치 흔적 제거(중복 방지) ---
    # initializer에서 mean/std 제거
    keep_inits = [init for init in g.initializer if init.name not in (MEAN_NAME, STD_NAME)]
    del g.initializer[:]
    g.initializer.extend(keep_inits)

    # Normalize 노드 제거
    keep_nodes = [node for node in g.node if node.name not in (SUB_NAME, DIV_NAME)]
    del g.node[:]
    g.node.extend(keep_nodes)

    # --- mean/std initializer 추가 ---
    g.initializer.extend([
        numpy_helper.from_array(mean.reshape(1, obs_dim), name=MEAN_NAME),
        numpy_helper.from_array(std.reshape(1, obs_dim),  name=STD_NAME),
    ])

    # obs_norm = (obs - mean) / std
    sub_node = helper.make_node("Sub", ["obs", MEAN_NAME], [OBS_CTR], name=SUB_NAME)
    div_node = helper.make_node("Div", [OBS_CTR, STD_NAME], [OBS_NORM], name=DIV_NAME)

    # 기존 그래프에서 'obs'를 쓰는 모든 노드 입력을 'obs_norm'으로 치환
    for node in g.node:
        for k, name in enumerate(node.input):
            if name == "obs":
                node.input[k] = OBS_NORM

    # 정규화 노드를 맨 앞에 삽입
    g.node.insert(0, div_node)
    g.node.insert(0, sub_node)

    onnx.checker.check_model(m)

    # 백업(최초 1회)
    backup = onnx_path.with_name("policy.nonorm.onnx")
    if not backup.exists():
        backup.write_bytes(onnx_path.read_bytes())

    onnx.save(m, str(onnx_path))
    return True, "patched exported/policy.onnx (norm included)"

def main():
    if not ROOT.is_dir():
        raise RuntimeError(f"ROOT not found: {ROOT}")

    run_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])

    ok = 0
    skip = 0
    for rd in run_dirs:
        success, msg = patch_policy_onnx(rd)
        print(("OK  " if success else "SKIP"), rd.name, ":", msg)
        ok += int(success)
        skip += int(not success)

    print(f"\nDone. OK={ok}, SKIP={skip}")
    print("Backups saved as policy.nonorm.onnx (only once per run).")

if __name__ == "__main__":
    main()