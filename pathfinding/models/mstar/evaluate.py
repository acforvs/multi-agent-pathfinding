import fire
import numpy as np
import os
import pickle

from pathfinding.environment import Environment
from pathfinding.models.mstar.call_cpp import prepare_cpp, call_single_mstar_test
from pathfinding.settings import yaml_data as settings
from pathfinding.utils import test_group, calculate_metrics

GENERAL_CONFIG = settings["dhc"]


def _prepare(filename="main.cpp"):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prepare_cpp(
        os.path.join(dir_path, filename), os.path.join(dir_path, ".", "main")
    )  # compiling the file with the C++ solution


def _test_one_case(args):
    map, agents_pos, goals_pos, actions = args
    env = Environment()

    env.load(map, agents_pos, goals_pos)
    _, _ = env.observe()

    done, steps = False, 0

    while not done and env.steps < GENERAL_CONFIG["max_episode_length"]:
        (_, _), _, done, _ = env.step([agent_action[steps] for agent_action in actions])
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
):
    def _test_generation_fn(tests):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        results = [
            call_single_mstar_test(test, os.path.join(dir_path, ".", "main"))
            for test in tests
        ]
        actions = [result[0] for result in results]
        return [(*test, action) for test, action in zip(tests, actions)]

    for group in test_groups:
        test_group(group, _test_generation_fn, _test_one_case)


if __name__ == "__main__":
    _prepare()
    fire.Fire()
