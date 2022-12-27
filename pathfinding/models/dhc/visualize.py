import fire
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import torch

from pathfinding.environment import Environment
from pathfinding.models.dhc import DHCNetwork
from pathfinding.utils import tests_dir_path

torch.manual_seed(239)
np.random.seed(239)
random.seed(239)
device = torch.device("cpu")
torch.set_num_threads(1)


def frametamer(imgs, env, init_img):
    imgs.append([])
    imgs[-1].append(init_img)
    for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
        zip(env.agents_pos, env.goals_pos)
    ):
        imgs[-1].append(
            plt.text(agent_y, agent_x, i, color="black", ha="center", va="center")
        )
        imgs[-1].append(
            plt.text(goal_y, goal_x, i, color="black", ha="center", va="center")
        )


def fill_map(env):
    map = np.copy(env.map)
    for agent_id in range(env.num_agents):
        x, y = env.agents_pos[agent_id], env.goals_pos[agent_id]
        if np.array_equal(x, y):
            map[tuple(x)] = 4
        else:
            map[tuple(x)] = 2
            map[tuple(y)] = 3
    map = map.astype(np.uint8)
    return map


def make_animation_single_text(
    model_id: int, test_name: str, test_case_idx: int = 0, steps: int = 256
):
    test_case_idx = int(test_case_idx)
    color_map = np.array(
        [
            [255, 255, 255],  # white
            [190, 190, 190],  # gray
            [0, 191, 255],  # blue
            [255, 165, 0],  # orange
            [0, 250, 154],  # green
        ]
    )

    network = DHCNetwork()
    network.eval()
    network.to(device)
    state_dict = torch.load(
        os.path.join(".", "models", f"{model_id}.pth"), map_location=device
    )
    network.load_state_dict(state_dict)

    with open(os.path.join(tests_dir_path(), test_name), "rb") as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])

    fig = plt.figure()

    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        map = fill_map(env)
        img = plt.imshow(color_map[map], animated=True)

        frametamer(imgs, env, img)

        actions, _, _, _ = network.step(
            torch.from_numpy(obs.astype(np.float32)).to(device),
            torch.from_numpy(pos.astype(np.float32)).to(device),
        )
        (obs, pos), _, done, _ = env.step(actions)

    if done and env.steps < steps:
        map = fill_map(env)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps - env.steps):
            frametamer(imgs, env, img)

    ani = animation.ArtistAnimation(
        fig, imgs, interval=600, blit=True, repeat_delay=1000
    )

    video_writer = animation.PillowWriter(fps=10)

    videos_dir = os.path.join(".", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    ani.save(
        os.path.join(videos_dir, f"{model_id}_{test_name}_{test_case_idx}.gif"),
        writer=video_writer,
    )


if __name__ == "__main__":
    fire.Fire(make_animation_single_text)
