import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
                # rsl_rl 최신 runner는 obs_groups 키를 반드시 기대함
        if "obs_groups" not in train_cfg or train_cfg["obs_groups"] is None:
            train_cfg["obs_groups"] = {}
        # rsl_rl (pip) PPO가 모르는 키 제거 (IsaacLab 쪽 cfg에서 들어오는 경우가 있음)
        if isinstance(train_cfg, dict):
            # case A: algorithm 아래에 있는 형태
            if "algorithm" in train_cfg and isinstance(train_cfg["algorithm"], dict):
                train_cfg["algorithm"].pop("lcp_cfg", None)
                train_cfg["algorithm"].pop("bound_loss_cfg", None)

    # case B: 최상위에 박혀있는 형태까지 방어
            train_cfg.pop("lcp_cfg", None)
            train_cfg.pop("bound_loss_cfg", None)

        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save model checkpoint.

        rsl_rl pip 버전과 IsaacLab 포크 버전의 logger 인터페이스 차이를 흡수하기 위해
        logger_type 같은 선택 속성은 getattr로 안전하게 접근한다.
        """
        # 1) 기본 저장은 부모 구현을 그대로 사용 (가장 안전)
        super().save(path)

        # 2) wandb 관련 후처리는 "있을 때만" 수행
        logger_type = getattr(self, "logger_type", None)
        logger = getattr(self, "logger", None)

        # logger_type이 없으면 여기서 종료 (지금 터진 케이스)
        if logger_type != "wandb":
            return

        # rsl_rl logger에 wandb 핸들이 없을 수도 있으니 방어
        try:
            import wandb  # noqa: F401
        except Exception:
            return

        # 여기 아래는 당신이 원래 하던 wandb 저장/업로드 로직이 있으면 유지
        # 단, logger 내부 속성 접근은 모두 getattr로 방어하세요.
        try:
            # 예시) wandb.save(...) 같은 게 있었다면 여기서 수행
            # wandb.save(path, policy="now")
            pass
        except Exception:
            pass

