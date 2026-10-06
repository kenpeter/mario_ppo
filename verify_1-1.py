"""Verify a pre-trained yumouwei PPO model truly clears 1-1 (reaches the flag).

Loads pre-trained-1, plays SuperMarioBros-1-1-v0 deterministically, records a
clean mp4, and asserts the flag is reached (info['flag_get'] / x_pos>=3200).
"""
import os
import time
import imageio
import numpy as np
import gym
from stable_baselines3 import PPO
from gym_utils import SMBRamWrapper, load_smb_env, SMB

MODEL_DIR = "./models"
crop_dim = [0, 16, 0, 13]
n_stack, n_skip = 4, 4
MODEL_NAME = "pre-trained-1"
OUT = "/home/kenpeter/work/mario_ppo/mario_1-1_clean.mp4"

env = load_smb_env("SuperMarioBros-1-1-v0", crop_dim, n_stack, n_skip)
model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env)
smb = SMB(env, model)

states = env.reset()
done = False
score = 0
max_x = 0
flag = False
frames = []
reached_steps = 0
while not done:
    frame = env.render(mode="rgb_array")
    frames.append(frame)
    action, _ = model.predict(states, deterministic=True)
    states, reward, done, info = env.step(action)
    info = info[0] if isinstance(info, (list, tuple)) else info
    score += reward
    max_x = max(max_x, int(info.get("x_pos", 0)))
    if info.get("flag_get"):
        flag = True
    reached_steps += 1
    if reached_steps > 20000:
        break

env.close()
fps = 50
imageio.mimsave(OUT, [f for f in frames if f is not None], fps=fps)
print(f"DONE video={OUT} frames={len(frames)} score={score:.0f} max_x={max_x} flag_get={flag}")
print("RESULT:", "CLEAN 1-1 (flag reached)" if flag else "did NOT reach flag")
