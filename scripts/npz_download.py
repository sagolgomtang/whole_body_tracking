import argparse
from pathlib import Path
import shutil
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="wandb-registry-motions")  # registry project
    parser.add_argument("--type", default="motions")
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    tmp_root = out_root / "_wandb_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    project_path = f"{args.entity}/{args.project}"
    atype = api.artifact_type(args.type, project_path)
    cols = list(atype.collections())
    print(f"[INFO] Found {len(cols)} motion collections in {project_path} (type={args.type}).")

    for i, col in enumerate(cols, 1):
        name = col.name
        full = f"{project_path}/{name}:latest"
        print(f"[{i}/{len(cols)}] Downloading: {full}")

        art = api.artifact(full)  # resolves :latest -> specific version
        # art.name is usually like "pushAndStumble1_subject5:v0"
        dest_dir = out_root / art.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # download into a unique temp dir to avoid overwriting
        tmp_dir = tmp_root / art.name
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        dl_dir = Path(art.download(root=str(tmp_dir)))

        # find motion.npz robustly (sometimes dl_dir == tmp_dir)
        motion_candidates = list(dl_dir.rglob("motion.npz"))
        if len(motion_candidates) == 0:
            raise FileNotFoundError(f"motion.npz not found under: {dl_dir}")

        motion_src = motion_candidates[0]
        motion_dst = dest_dir / "motion.npz"
        shutil.copy2(motion_src, motion_dst)

        print(f"      -> {motion_dst}")

    print("[DONE] All motions downloaded.")


if __name__ == "__main__":
    main()
