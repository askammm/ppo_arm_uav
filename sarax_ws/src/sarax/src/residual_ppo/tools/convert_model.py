#!/usr/bin/env python3
"""
convert_model.py
================
在宿主机（NumPy 2.x）运行，将 SB3 PPO 模型中的 actor 网络导出为
TorchScript (.pt)，消除 numpy._core 依赖，使其可在旧版 numpy 1.x
的容器内无缝加载。

同时将 VecNormalize 统计量（obs_rms.mean / var）导出为纯 numpy .npz，
同样不依赖 SB3 或 numpy 版本。

用法（宿主机）:
    cd sarax_ws/src/sarax/src/residual_ppo
    python tools/convert_model.py \
        models/ppo_residual_sarax_final \
        [--vecnorm models/ppo_residual_sarax_vecnorm.pkl] \
        [--output  models/actor.pt]

输出:
    models/actor.pt          — TorchScript actor（只含推理部分）
    models/obs_rms.npz       — VecNormalize 观测归一化统计（可选）
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Actor-only wrapper（只保留推理所需的三层：特征提取→策略MLP→动作头）
# ---------------------------------------------------------------------------

class ActorOnly(nn.Module):
    """
    提取 SB3 PPO ActorCriticPolicy 中的 actor 路径：
      obs -> pi_features_extractor -> mlp_extractor.policy_net -> action_net
    推理结果是未裁剪的 tanh 前 mean action（SB3 PPO 默认不用 tanh squash）。
    调用方负责 clip 到 [-1, 1]。
    """
    def __init__(self, policy):
        super().__init__()
        self.feat_ext   = policy.pi_features_extractor   # FlattenExtractor
        self.policy_net = policy.mlp_extractor.policy_net  # nn.Sequential
        self.action_net = policy.action_net               # nn.Linear

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features   = self.feat_ext(obs)
        latent_pi  = self.policy_net(features)
        return self.action_net(latent_pi)


# ---------------------------------------------------------------------------
# 主转换逻辑
# ---------------------------------------------------------------------------

def convert(model_path: str, vecnorm_path: str | None, output_pt: str):
    from stable_baselines3 import PPO

    zip_path = model_path if model_path.endswith(".zip") else model_path + ".zip"
    if not os.path.exists(zip_path):
        sys.exit(f"[convert] 找不到模型文件: {zip_path}")

    print(f"[convert] 载入 SB3 模型: {zip_path}")
    model = PPO.load(model_path, device="cpu")
    policy = model.policy
    policy.eval()

    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]
    print(f"[convert] obs_dim={obs_dim}  act_dim={act_dim}")

    # ── 导出 actor 为 TorchScript ──
    actor = ActorOnly(policy)
    actor.eval()
    dummy = torch.zeros(1, obs_dim)

    try:
        traced = torch.jit.trace(actor, dummy)
        # 验证输出形状
        out = traced(dummy)
        assert out.shape == (1, act_dim), f"输出形状异常: {out.shape}"
        torch.jit.save(traced, output_pt)
        print(f"[convert] TorchScript actor 已保存: {output_pt}")
    except Exception as e:
        sys.exit(f"[convert] TorchScript trace 失败: {e}\n"
                 "建议检查策略网络结构是否含动态控制流。")

    # ── 导出 VecNormalize 统计量（可选）──
    if vecnorm_path and os.path.exists(vecnorm_path):
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        # 构造最小 dummy env 以便 VecNormalize 能 load
        import gymnasium as gym

        def _make():
            return gym.make("Pendulum-v1")   # 任意 env，只用来获取 shape 容器

        try:
            # 直接用 pickle 读取，避免构造 env
            import pickle
            with open(vecnorm_path, "rb") as f:
                vn_obj = pickle.load(f)
            # vn_obj 可能是 VecNormalize 实例或 RunningMeanStd 的字典
            # SB3 VecNormalize.save() 实际保存的是整个对象
            obs_rms = vn_obj.obs_rms
            npz_path = os.path.join(os.path.dirname(output_pt), "obs_rms.npz")
            np.savez(npz_path,
                     mean=obs_rms.mean.astype(np.float32),
                     var=obs_rms.var.astype(np.float32),
                     count=np.array([obs_rms.count], dtype=np.float64))
            print(f"[convert] obs_rms 已保存: {npz_path}")
        except Exception as e:
            print(f"[convert] 警告: VecNormalize 导出失败 ({e})，跳过。")
    else:
        print("[convert] 未提供 vecnorm 路径，跳过归一化统计导出。")

    print("[convert] 完成。")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="将 SB3 PPO (numpy 2.x 保存) 转换为 TorchScript，"
                    "消除 numpy._core 依赖。")
    ap.add_argument("model",
                    help="SB3 模型路径（有无 .zip 后缀均可）")
    ap.add_argument("--vecnorm", default=None,
                    help="VecNormalize .pkl 路径（可选）")
    ap.add_argument("--output", default=None,
                    help="输出 .pt 路径（默认: <model_dir>/actor.pt）")
    args = ap.parse_args()

    model_path = args.model.removesuffix(".zip")
    output_pt  = args.output or os.path.join(
        os.path.dirname(os.path.abspath(model_path)), "actor.pt")

    convert(model_path, args.vecnorm, output_pt)


if __name__ == "__main__":
    main()
