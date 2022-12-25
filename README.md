# Multi-agent Path Finding using Reinforcement Learning


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-%2300C4CC.svg?style=flat&logo=Poetry&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Setting up
1. Install [Poetry](https://python-poetry.org)
2. Run [poetry install](https://python-poetry.org/docs/cli/#install) to install the dependencies

If you see ``Failed to create the collection: Prompt dismissed..`` this error when trying to run `poetry install`, consider executing this line first:
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

## Requirements
In order for `models.dhc.train` to be successfully run, you have to have a machine equipped with 1 GPU and several CPUs.
Consider having `num_cpus - 2` actors configured through the `dhc.train.num_actors` in `config.yaml`

**Attention: We do not guarantee the desired performance on a non-GPU machine.**


## Contributing
1. Use [black](https://github.com/psf/black) to ensure that the codestyle remains great
```shell
poetry run black dir
```
2. Make sure tests are OK 
```shell
poetry run pytest
```
3. Create a PR with new features


## References

<a id="1">[1]</a> 
Sartoretti, G., Kerr, J., Shi, Y., Wagner, G., Kumar, T.S., Koenig, S. and Choset, H., 2019. Primal: Pathfinding via reinforcement and imitation multi-agent learning. IEEE Robotics and Automation Letters, 4(3), pp.2378-2385.

<a id="2">[2]</a> 
Ma, Ziyuan and Luo, Yudong and Ma, Hang, 2021. Distributed Heuristic Multi-Agent Path Finding with Communication.

<a id="3">[3]</a>
Skrynnik, Alexey and Andreychuk, Anton and Yakovlev, Konstantin and Panov, Aleksandr I., 2022. POGEMA: Partially Observable Grid Environment for Multiple Agents. 


## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/acforvs/multi-agent-pathfinding/blob/main/LICENSE)


