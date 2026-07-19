import gym
import numpy as np
import shutil
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import stable_baselines3
import torch
from r3m import load_r3m
from gym_ras.api import make_env
import os
from pathlib import Path
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path


class MyCheckpointCallback(CheckpointCallback):
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print("")
            print(
                f"=============timestep {self.num_timesteps} =============================")
            print("")
            path = os.path.join(self.save_path, f"{self.name_prefix}")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


class ProgressCallback(BaseCallback):
    """heartbeat print so long silent stretches are visible in the terminal"""

    def __init__(self, print_freq=100, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self._t0 = None
        self._last_t = None
        self._last_step = None

    # NOTE: BaseCallback.num_timesteps only syncs from the model inside
    # on_step(); in training/rollout hooks it is stale (0 on resume),
    # so always read self.model.num_timesteps here
    def _on_training_start(self) -> None:
        self._t0 = time.time()
        self._last_t = self._t0
        self._last_step = self.model.num_timesteps
        print(f"[train] start, timestep {self.model.num_timesteps}")

    def _on_rollout_start(self) -> None:
        print(f"[train] collecting rollout ({self.model.n_steps} steps) ..., timestep {self.model.num_timesteps}")

    def _on_rollout_end(self) -> None:
        print(f"[train] rollout done, updating policy ..., timestep {self.model.num_timesteps}")

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            now = time.time()
            fps = (self.num_timesteps - self._last_step) / \
                max(now - self._last_t, 1e-8)
            print(f"[train] timestep {self.num_timesteps} | {fps:.2f} steps/s | elapsed {(now - self._t0) / 60.0:.1f} min")
            self._last_t = now
            self._last_step = self.num_timesteps
        return True


class Env:

    metadata = {}

    def __init__(self, env):
        # self.env = embodied.envs.load_single_env('ur5_real', length=100)
        self.env = env
        self.num_actions = self.env.action_space.n
        self.score = 0
        self.length = 0
        self.r3m = load_r3m("resnet50")
        self.r3m.cuda()
        self.r3m.eval()

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self.env.observation_space.items():
            if key.startswith('log_'):
                continue
            if key.startswith('is_'):
                continue
            if key in ('reward', 'depth'):
                continue
            if key == 'image':
                spaces[key] = gym.spaces.Box(
                    -np.inf, np.inf, (2048,), np.float)
                continue

            spaces[key] = gym.spaces.Box(
                value.low, value.high, value.shape, value.dtype)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.num_actions)

    def reset(self):
        obs = self.env.reset()
        with torch.no_grad():
            image = torch.tensor(obs['image'].copy()).cuda()
            obs['image'] = self.r3m(image.permute(2, 0, 1)[None])[
                0].cpu().numpy()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        with torch.no_grad():
            image = torch.tensor(obs['image'].copy()).cuda()
            obs['image'] = self.r3m(image.permute(2, 0, 1)[None])[
                0].cpu().numpy()
        return obs, reward, done, info

    def _stack_obs(self, obs):
        obs['image']


def eval_ppo(env, config, n_eval_episodes, save_prefix):
    _dir = config.logdir
    env = Env(env)
    # NOTE: PPO.load is a classmethod, it returns a new model;
    # calling model.load() in-place does NOT load weights
    model = PPO.load(str(Path(config.logdir) / "best_model.zip"),
                     env=env, verbose=10,
                     tensorboard_log=_dir,
                     print_system_info=True)
    vec_env = DummyVecEnv([lambda: env])
    eval_stat = {'success_eps': 0, 'success_rate': 0,
                 'total_eps': 0, "score": []}
    max_eps_length = 300

    import yaml as yl
    from pathlib import Path
    from datetime import datetime
    _dir = Path("./data") / "exp_result"
    _dir.mkdir(parents=True, exist_ok=True)
    _file = _dir / (save_prefix + "@seed" + str(env.env.seed) + "@" +
                    str(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")) + ".yml")

    for j in range(n_eval_episodes):
        obs = vec_env.reset()
        done = False
        eps_length = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            eps_length += 1
            done = dones[0]
            print(eps_length)
        if info[0]['fsm'] == "done_success":
            eval_stat['success_eps'] += 1
            print("+++sucess episode !")
            score = (max_eps_length - eps_length) / max_eps_length
        else:
            score = 0

        eval_stat['total_eps'] += 1
        
        eval_stat['score'].append(score)
        eval_stat['success_rate'] = eval_stat['success_eps'] / \
            eval_stat['total_eps']
        print(eval_stat)
        with open(str(_file), 'w') as yaml_file:
            yl.dump(eval_stat, yaml_file, default_flow_style=False)

    eval_stat["score_mean"] = np.mean(np.array(eval_stat["score"])).item(0)
    eval_stat["score_std"] = np.std(np.array(eval_stat["score"])).item(0)
    return eval_stat


def _backup_reload_files(logdir):
    # resuming overwrites best_model.zip / evaluations.npz (EvalCallback
    # restarts with best_mean_reward=-inf and clobbers eval history) and
    # checkpoint.zip; keep a copy tagged with a unique id per resume
    backup_id = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    for fname in ("best_model.zip", "evaluations.npz", "checkpoint.zip"):
        f = Path(logdir) / fname
        if f.is_file():
            dst = f.with_name(f"{f.stem}@bak-{backup_id}{f.suffix}")
            shutil.copy2(str(f), str(dst))
            print(f"backup {fname} -> {dst.name}")


def train(env, config, is_reload=False, only_eval=False):
    _dir = config.logdir
    env = Env(env)
    if is_reload:
        _backup_reload_files(config.logdir)
        # prefer the latest checkpoint over best_model for resuming
        ckpt = Path(config.logdir) / "checkpoint.zip"
        if not ckpt.is_file():
            ckpt = Path(config.logdir) / "best_model.zip"
        # NOTE: PPO.load is a classmethod, it returns a new model;
        # calling model.load() in-place does NOT load weights.
        # num_timesteps is restored from the checkpoint, so the
        # oracle/random prefill stage will not re-trigger on resume.
        model = PPO.load(str(ckpt), env=env, verbose=1,
                         tensorboard_log=_dir,
                         print_system_info=True)
        print(f"reload from {ckpt}, resume at timestep {model.num_timesteps}")
    else:
        model = PPO(
            stable_baselines3.common.policies.MultiInputActorCriticPolicy,
            env, verbose=1,
            tensorboard_log=_dir
        )
    model._prefill_oracle = config.prefill_oracle
    model._prefill_random = config.prefill_random

    eval_callback = EvalCallback(env, best_model_save_path=_dir,
                                 log_path=_dir if not only_eval else str(
                                     Path(config.logdir) / "online_results"),
                                 n_eval_episodes=config.eval_eps,
                                 eval_freq=config.eval_freq,
                                 deterministic=True,
                                 render=False)
    cbs = [eval_callback, ProgressCallback(print_freq=100)]
    if not only_eval:
        # old reload dirs' baseline_config.yaml may not have ckpt_freq
        ckpt_freq = config.ckpt_freq if 'ckpt_freq' in config else 1e4
        checkpoint_cb = MyCheckpointCallback(
            save_freq=ckpt_freq, save_path=_dir, name_prefix="checkpoint", verbose=1,)
        cbs.append(checkpoint_cb)


# if __name__  == "__main__":
#     env, env_config = make_env(tags=["dvrk_cam_setting","grasp_any"], seed=0)
#     _dir = "./log_ppo/"
#     env = Env(env)
#     model = stable_baselines3.PPO(
#         stable_baselines3.common.policies.MultiInputActorCriticPolicy,
#         env, verbose=1,
#         tensorboard_log=_dir
#         )

    # eval_callback = EvalCallback(env, best_model_save_path=_dir,
    #                              log_path=_dir,
    #                              #  eval_freq=3e4,
    #                              eval_freq=1e2,
    #                              deterministic=True,
    #                              render=False)

    # reset_num_timesteps=False on reload: keep the restored step counter so
    # tensorboard/checkpoint steps continue and prefill does not re-trigger
    model.learn(total_timesteps=1e6, callback=cbs,
                reset_num_timesteps=not is_reload)
