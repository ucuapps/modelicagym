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

## Examples
Examples of usage of both packages can be found in examples folder.
* [cart_pole_env.py](https://github.com/OlehLuk/modelica-gym/blob/master/examples/cart_pole_env.py) 
is an example how a specific FMU can be integrated to an OpenAI Gym as an environment. Classic cart-pole environment is considered. 
Corresponding FMU's can be found in the resources folder.

* [cart_pole_q_learner.py](https://github.com/OlehLuk/modelica-gym/blob/master/examples/cart_pole_q_learner.py) 
is an example of Q-learning algorithm application. Agent is trained on the Cart-pole environment simulated with an FMU. Its' integration is described in previous example.
