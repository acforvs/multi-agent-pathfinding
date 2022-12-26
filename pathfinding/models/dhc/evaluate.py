import os
import fire
import numpy as np
import pickle

from pathfinding.models.dhc import DHCNetwork
from pathfinding.environment import Environment

from pathfinding.settings import yaml_data as settings


GENERAL_CONFIG = settings["dhc"]


def test_one_case(map, agents_pos, goals_pos, network):
    env = Environment()

    env.load(map, agents_pos, goals_pos)
    obs, pos = env.observe()

    done, steps = False, 0
    network.reset()

    while not done and env.steps < GENERAL_CONFIG["max_episode_length"]:
        actions, _, _, _ = network.step(
            torch.as_tensor(obs.astype(np.float32)),
            torch.as_tensor(pos.astype(np.float32)),
        )
        (obs, pos), _, done = env.step(actions)
        steps += 1

    return np.array_equal(env.agents_pos, env.goals_pos), steps


def _generate_test_filename(length: int, num_agents: int, density: float, ext="pkl"):
    return f"{length}length_{num_agents}agents_{density}density.{ext}"


def _tests_dir_path():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "test_cases")


def generate_test_suits(tests_config, repeat_for: int):
    os.makedirs(_tests_dir_path(), exist_ok=True)
    for map_length, num_agents, density in tests_config:
        env = Environment(
            num_agents=num_agents, map_length=map_length, fix_density=density
        )
        tests = []
        for _ in range(repeat_for):
            tests.append(
                (np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos))
            )
            env.reset(num_agents=num_agents, map_length=map_length)

        filename = _generate_test_filename(map_length, num_agents, density)
        with open(os.path.join(_tests_dir_path(), filename), "wb") as file:
            pickle.dump(tests, file)


if __name__ == "__main__":
    fire.Fire()
