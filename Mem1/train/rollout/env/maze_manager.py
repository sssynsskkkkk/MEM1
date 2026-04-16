from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from .maze import MazeEnv


class MazeEnvManager:
    def __init__(self):
        self.envs: Dict[int, MazeEnv] = {}

    def reset(self, environment_ids: Iterable[int], maze_specs: Sequence[Dict]) -> List[str]:
        observations: List[str] = []
        for environment_id, maze_spec in zip(environment_ids, maze_specs):
            env = MazeEnv(maze_spec=maze_spec)
            self.envs[int(environment_id)] = env
            observations.append(env.reset())
        return observations

    def step(self, actions: Sequence[str], environment_ids: Sequence[int]):
        observations, rewards, dones, infos = [], [], [], []
        for environment_id, action in zip(environment_ids, actions):
            env = self.envs[int(environment_id)]
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return observations, rewards, dones, infos
