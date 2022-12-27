import fire
import multiprocessing as mp
import numpy as np
import os
import pickle

from pathfinding.environment import Environment


def generate_test_filename(length: int, num_agents: int, density: float, ext="pkl"):
    return f"{length}length_{num_agents}agents_{density}density.{ext}"


def tests_dir_path():
    return os.path.join(".", "pathfinding", "test_cases")


def generate_test_suits(tests_config, repeat_for: int):
    os.makedirs(tests_dir_path(), exist_ok=True)
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

        filename = generate_test_filename(map_length, num_agents, density)
        with open(os.path.join(tests_dir_path(), filename), "wb") as file:
            pickle.dump(tests, file)


def test_group(test_group, test_generation_fn, singe_test_fn):
    pool = mp.Pool(mp.cpu_count())

    length, num_agents, density = test_group

    print(f"test group: {length} length {num_agents} agents {density} density")

    with open(
        os.path.join(
            tests_dir_path(),
            generate_test_filename(length, num_agents, density),
        ),
        "rb",
    ) as f:
        tests = pickle.load(f)
    tests = test_generation_fn(tests)
    ret = pool.map(singe_test_fn, tests)

    success, avg_step, soft_equality = 0, 0, 0
    for is_successful, ttl_steps, soft_equal in ret:
        success += is_successful
        avg_step += ttl_steps
        soft_equality += soft_equal

    print(f"success rate: {success/len(ret) * 100:.2f}%")
    print(f"soft-success rate: {soft_equality / len(ret) * 100:.2f}%")
    print(f"average step: {avg_step/len(ret)}")
    print()


def calculate_metrics(env: Environment, steps: int):
    pos_equality = env.agents_pos == env.goals_pos
    soft_equality = (
        pos_equality[:, 0] * pos_equality[:, 1]
    ).sum() / env.agents_pos.shape[0]

    return np.array_equal(env.agents_pos, env.goals_pos), steps, soft_equality


if __name__ == "__main__":
    fire.Fire()
