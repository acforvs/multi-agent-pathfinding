import numpy as np
import os
import random
import ray
import time
import torch

from pathfinding.models.dhc import GlobalBuffer, Learner, Actor
from pathfinding.settings import yaml_data as settings

TRAIN_CONFIG = settings["dhc"]["train"]

torch.manual_seed(239)
np.random.seed(239)
random.seed(239)


def main(
    num_actors=TRAIN_CONFIG["num_actors"], log_interval=TRAIN_CONFIG["log_interval"]
):
    ray.init()
    ray_node = ray.nodes()[0]

    # GlobalBuffer + Learner + 1 * num_actors
    assert (
        ray_node["Resources"]["CPU"] >= 2 + num_actors
    ), "insufficient amount of CPU cores available"

    buffer = GlobalBuffer.remote()
    learner = Learner.remote(buffer)
    time.sleep(1)
    actors = [
        Actor.remote(i, 0.4 ** (1 + (i / (num_actors - 1)) * 7), learner, buffer)
        for i in range(num_actors)
    ]

    for actor in actors:
        actor.run.remote()

    print("Actors were successfully created")

    while not ray.get(buffer.ready.remote()):
        time.sleep(5)
        ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))

    print("Start training")
    buffer.run.remote()
    learner.run.remote()

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))
        print()


if __name__ == "__main__":
    main()
