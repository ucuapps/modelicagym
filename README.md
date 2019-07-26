# modelica-gym
Repository contains:
* `modelicagym.environments` package for integration of FMU as an environment to OpenAI Gym.
FMU is a functional model unit exported from one of the main Modelica tools, e.g. Dymola(proprietary) or JModelica(open source).
Currently only FMU's exported in co-simulation mode are supported.
* `gymalgs.rl` package for Reinforcement Learning algorithms compatible to OpenAI Gym environments.

## Instalation
Full instalation guide is available [here](https://github.com/OlehLuk/modelica-gym/blob/master/docs/install.md).

You can test working environment by running 
[./test_setup.py](https://github.com/OlehLuk/modelica-gym/blob/master/test/setup_test.py) script.
