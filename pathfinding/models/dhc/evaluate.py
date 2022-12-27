import fire
import numpy as np
import os
import pickle
import torch

from pathfinding.environment import Environment
from pathfinding.models.dhc import DHCNetwork
from pathfinding.settings import yaml_data as settings
from pathfinding.utils import test_group, calculate_metrics

GENERAL_CONFIG = settings["dhc"]


def _test_one_case(args):
    map, agents_pos, goals_pos, network = args
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
        (obs, pos), _, done, _ = env.step(actions)
        steps += 1

    return calculate_metrics(env, steps)


def test_model(
    test_groups=[
        (40, 4, 0.3),
        (40, 8, 0.3),
        (40, 16, 0.3),
        (40, 32, 0.3),
        (40, 64, 0.3),
        (80, 4, 0.3),
        (80, 8, 0.3),
        (80, 16, 0.3),
        (80, 32, 0.3),
        (80, 64, 0.3),
    ],
    model_number=60000,
):
    network = DHCNetwork()
    network.eval()
    device = torch.device("cpu")
    network.to(device)
    state_dict = torch.load(
        os.path.join(".", "models", f"{model_number}.pth"), map_location=device
    )
    network.load_state_dict(state_dict)
    network.eval()
    network.share_memory()

    def _test_generation_fn(tests):
        return [(*test, network) for test in tests]

    for group in test_groups:
        test_group(group, _test_generation_fn, _test_one_case)


if __name__ == "__main__":
    fire.Fire()
