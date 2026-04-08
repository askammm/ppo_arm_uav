#!/usr/bin/env python3
"""
train_ppo.py
Train (or fine-tune) the residual PPO policy using Stable-Baselines3.

── Pretraining ───────────────────────────────────────────────────────────
  python train/train_ppo.py --config train/config.yaml

── Fine-tuning from a pretrained checkpoint ─────────────────────────────
  python train/train_ppo.py --config train/config.yaml \
      --resume models/ppo_residual_sarax_final

  When --resume is used:
    • Loads the existing model weights (network architecture preserved)
    • Loads VecNormalize stats so obs normalisation is consistent
    • Switches to finetune: hyperparams from config (lower LR, tighter clip)
    • Starts arm curriculum from finetune.arm_start_mode (skips static stage)
    • Continues wind curriculum from finetune.wind_start

Requirements:
    pip install stable-baselines3[extra] gymnasium pyyaml
"""

import argparse
import os
import sys
import yaml
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from envs.residual_env import ResidualPPOEnv


# ---------------------------------------------------------------------------
# Helper: reach into VecEnv layers to get the raw Gym envs
# ---------------------------------------------------------------------------

def _get_raw_envs(vec_env):
    """Recursively unwrap VecNormalize/VecEnv to get list of base envs."""
    env = vec_env
    while hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs"):
        return env.envs
    return []


# ---------------------------------------------------------------------------
# Callback: wind curriculum
# ---------------------------------------------------------------------------

class WindCurriculumCallback(BaseCallback):
    """Linearly ramp wind from wind_start→wind_max over anneal_steps."""

    def __init__(self, wind_start, wind_max, anneal_steps,
                 start_timestep=0, verbose=0):
        super().__init__(verbose)
        self.wind_start      = wind_start
        self.wind_max        = wind_max
        self.anneal_steps    = anneal_steps
        self.start_timestep  = start_timestep  # offset for fine-tune resume

    def _on_step(self):
        effective_steps = self.num_timesteps + self.start_timestep
        frac = min(1.0, effective_steps / max(1, self.anneal_steps))
        wind = self.wind_start + frac * (self.wind_max - self.wind_start)
        for e in _get_raw_envs(self.training_env):
            if hasattr(e, "set_wind_curriculum"):
                e.set_wind_curriculum(wind)
        return True


# ---------------------------------------------------------------------------
# Callback: arm curriculum
# ---------------------------------------------------------------------------

class ArmCurriculumCallback(BaseCallback):
    """
    Progressive arm motion curriculum:
      Step 0           → transition_steps    : "static"
      transition_steps → transition_dynamic  : "slow"
      transition_dynamic → end               : "dynamic"

    For fine-tuning with start_mode != "static", the first transition
    is skipped entirely.
    """

    # Map stage name → numeric level for comparison
    _STAGE_LEVEL = {"static": 0, "slow": 1, "dynamic": 2}

    def __init__(self, stage1, stage2, stage3,
                 transition_steps, transition_steps_dynamic,
                 start_mode="static", verbose=0):
        super().__init__(verbose)
        self.stages = [stage1, stage2, stage3]
        self.t1 = transition_steps
        self.t2 = transition_steps_dynamic
        self._start_level = self._STAGE_LEVEL.get(start_mode, 0)
        self._current_stage = None

    def _set_stage(self, stage: str):
        if stage == self._current_stage:
            return
        self._current_stage = stage
        if self.verbose:
            print(f"[ArmCurriculum] switching to '{stage}' "
                  f"at step {self.num_timesteps:,}")
        for e in _get_raw_envs(self.training_env):
            if hasattr(e, "set_arm_curriculum"):
                e.set_arm_curriculum(stage)

    def _on_step(self):
        t = self.num_timesteps
        # Determine target stage
        if t < self.t1 and self._start_level <= 0:
            target = self.stages[0]   # "static"
        elif t < self.t2 and self._start_level <= 1:
            target = self.stages[1]   # "slow"
        else:
            target = self.stages[2]   # "dynamic"

        # If fine-tune starts at "slow", jump straight past static
        if self._start_level >= 1 and target == self.stages[0]:
            target = self.stages[1]
        if self._start_level >= 2:
            target = self.stages[2]

        self._set_stage(target)
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_env_fn(env_cfg=None):
    def _init():
        return ResidualPPOEnv(env_cfg)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=os.path.join(_ROOT, "train", "config.yaml"))
    parser.add_argument("--resume", default=None,
                        help="Path to model zip (without .zip) to resume/fine-tune")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    ppo_cfg  = cfg["ppo"]
    cur_cfg  = cfg.get("curriculum", {})
    arm_cfg  = cfg.get("arm_curriculum", {})
    ft_cfg   = cfg.get("finetune", {})
    paths    = cfg.get("paths", {})

    is_finetune = (args.resume is not None
                   and os.path.exists(args.resume + ".zip"))

    model_dir  = os.path.join(_ROOT, paths.get("model_save_dir", "models"))
    log_dir    = os.path.join(_ROOT, paths.get("log_dir", "logs"))
    model_name = paths.get("model_name", "ppo_residual_sarax")
    if is_finetune:
        model_name = model_name + "_ft"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)

    n_envs = ppo_cfg.get("n_envs", 8)

    # ── Build environments ────────────────────────────────────────────
    env = make_vec_env(make_env_fn(), n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(make_env_fn(), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    # ── Load or create model ──────────────────────────────────────────
    if is_finetune:
        print(f"[train_ppo] Fine-tuning from: {args.resume}.zip")

        # Load VecNormalize stats so obs normalisation is consistent
        vecnorm_path = os.path.join(model_dir,
                                    paths.get("model_name", "ppo_residual_sarax")
                                    + "_vecnorm.pkl")
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env)
            env.training    = True
            env.norm_reward = True
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training    = False
            eval_env.norm_reward = False
            print(f"[train_ppo] Loaded VecNormalize stats: {vecnorm_path}")
        else:
            print("[train_ppo] WARNING: vecnorm.pkl not found, "
                  "starting fresh normalisation (may cause instability)")

        model = PPO.load(args.resume, env=env, device="cpu")

        # Override hyper-parameters for fine-tuning
        model.learning_rate = float(ft_cfg.get("learning_rate",
                                                ppo_cfg["learning_rate"]))
        model.clip_range    = lambda _: float(ft_cfg.get("clip_range",
                                                          ppo_cfg["clip_range"]))
        model.ent_coef      = float(ft_cfg.get("ent_coef",
                                               ppo_cfg["ent_coef"]))
        print(f"[train_ppo] Fine-tune LR={model.learning_rate:.2e}  "
              f"clip={ft_cfg.get('clip_range', ppo_cfg['clip_range'])}  "
              f"ent={model.ent_coef:.4f}")

        total_timesteps = int(ft_cfg.get("total_timesteps",
                                          ppo_cfg["total_timesteps"]))
        arm_start_mode  = ft_cfg.get("arm_start_mode", "slow")
        wind_start_val  = float(ft_cfg.get("wind_start",
                                            cur_cfg.get("wind_start", 0.0)))
        wind_max_val    = float(ft_cfg.get("wind_max",
                                            cur_cfg.get("wind_max", 5.0)))
        reset_timesteps = False   # keep global step counter for logging

    else:
        print("[train_ppo] Starting fresh pretraining")
        policy_kwargs = {}
        if "policy_kwargs" in ppo_cfg and "net_arch" in ppo_cfg["policy_kwargs"]:
            policy_kwargs["net_arch"] = ppo_cfg["policy_kwargs"]["net_arch"]

        model = PPO(
            policy=ppo_cfg.get("policy", "MlpPolicy"),
            env=env,
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 256),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.005),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
            policy_kwargs=policy_kwargs or None,
            tensorboard_log=log_dir,
            verbose=1,
        )
        total_timesteps = int(ppo_cfg.get("total_timesteps", 2_000_000))
        arm_start_mode  = "static"
        wind_start_val  = float(cur_cfg.get("wind_start", 0.0))
        wind_max_val    = float(cur_cfg.get("wind_max", 5.0))
        reset_timesteps = True

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = []

    # Checkpoint
    callbacks.append(CheckpointCallback(
        save_freq=max(1, 100_000 // n_envs),
        save_path=model_dir,
        name_prefix=model_name,
    ))

    # Evaluation
    callbacks.append(EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(1, paths.get("eval_freq", 50_000) // n_envs),
        n_eval_episodes=paths.get("n_eval_episodes", 20),
        deterministic=True,
        render=False,
    ))

    # Wind curriculum
    if cur_cfg.get("enabled", True):
        callbacks.append(WindCurriculumCallback(
            wind_start=wind_start_val,
            wind_max=wind_max_val,
            anneal_steps=int(cur_cfg.get("anneal_steps", 1_500_000)),
            start_timestep=0 if reset_timesteps else model.num_timesteps,
        ))

    # Arm curriculum
    if arm_cfg.get("enabled", True):
        callbacks.append(ArmCurriculumCallback(
            stage1=arm_cfg.get("stage1_mode", "static"),
            stage2=arm_cfg.get("stage2_mode", "slow"),
            stage3=arm_cfg.get("stage3_mode", "dynamic"),
            transition_steps=int(arm_cfg.get("transition_steps", 600_000)),
            transition_steps_dynamic=int(
                arm_cfg.get("transition_steps_dynamic", 1_200_000)),
            start_mode=arm_start_mode,
            verbose=1,
        ))

    # ── Train ─────────────────────────────────────────────────────────
    print(f"[train_ppo] {'Fine-tuning' if is_finetune else 'Training'} "
          f"for {total_timesteps:,} steps  "
          f"arm_start={arm_start_mode}  "
          f"wind={wind_start_val:.1f}→{wind_max_val:.1f} N")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
    )

    # ── Save ──────────────────────────────────────────────────────────
    final_path = os.path.join(model_dir, model_name + "_final")
    model.save(final_path)
    env.save(os.path.join(model_dir, model_name + "_vecnorm.pkl"))
    print(f"[train_ppo] Saved → {final_path}.zip")


if __name__ == "__main__":
    main()
