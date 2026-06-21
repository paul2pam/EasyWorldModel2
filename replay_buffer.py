import numpy as np
from collections import deque


class ReplayBuffer:
    """Stores episodes and samples contiguous sequences for RSSM training."""

    def __init__(self, max_episodes=200):
        self.episodes = deque(maxlen=max_episodes)
        self._cur_obs = []
        self._cur_actions = []

    def start_episode(self):
        self._cur_obs = []
        self._cur_actions = []

    def add(self, obs, action):
        # obs: (3, 64, 64) float32 in [0,1]; action: (action_dim,) float32
        self._cur_obs.append(np.array(obs, dtype=np.float32))
        self._cur_actions.append(np.array(action, dtype=np.float32))

    def end_episode(self):
        if len(self._cur_obs) > 1:
            self.episodes.append({
                "obs":     np.stack(self._cur_obs),     # (T, 3, 64, 64)
                "actions": np.stack(self._cur_actions), # (T, action_dim)
            })

    def sample(self, batch_size, seq_len):
        # seq_len: number of consecutive REAL frames per training sequence.
        # This is NOT imagination depth — the model trains on real obs the whole time.
        # Longer = better temporal credit assignment, but more memory. ~50 is typical.
        obs_batch, act_batch = [], []
        valid = [ep for ep in self.episodes if len(ep["obs"]) >= seq_len]
        if not valid:
            raise RuntimeError(f"No episodes long enough to sample seq_len={seq_len}. Collect more data first.")
        for _ in range(batch_size):
            ep = valid[np.random.randint(len(valid))]
            T = len(ep["obs"])
            start = np.random.randint(0, T - seq_len + 1)
            obs_batch.append(ep["obs"][start:start + seq_len])
            act_batch.append(ep["actions"][start:start + seq_len])
        return np.stack(obs_batch), np.stack(act_batch)  # (B, T, ...) each

    def __len__(self):
        return len(self.episodes)
