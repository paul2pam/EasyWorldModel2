import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F


class CarRacingWrapper(gym.Wrapper):
    """Resize CarRacing-v2 obs from 96x96 to 64x64 and normalize to [0, 1]."""

    def __init__(self, size=64):
        env = gym.make("CarRacing-v2", continuous=True)
        super().__init__(env)
        self.size = size

    def _preprocess(self, obs):
        # obs: (96, 96, 3) uint8 -> (3, 64, 64) float32 in [0, 1]
        t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return t.squeeze(0)  # (3, 64, 64)

    def reset(self, **kwargs): #resets entire env
        obs, info = self.env.reset(**kwargs)
        return self._preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._preprocess(obs), reward, terminated, truncated, info
