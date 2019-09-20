# How to use `modelicagym`? 

This file describes in a step-wise manner 
how to integrate your FMU with OpenAI Gym as an environment.

Instructions are illustrated with an example of cart-pole environment 
simulated by FMU exported in co-simulation mode from JModelica (FMI standard v.2.0). It can be found in resources folder:
`resources/jmodelica/linux/ModelicaGym_CartPole_CS.fmu`

## Prerequisites

Instructions assume that you have environment setup according to installation guide
including:
* OpenAI Gym installed.
* **working** PyFMI installation (requires Assimulo and Sundials libraries).

If you run *test/test_setup.py* and no errors occur, everything should be fine.

For sure, you should also have FMU of your model 
exported from some of the Modelica tools. 
Currently, only FMU's exported in co-simulation mode were tested. 

 
## 1. Understanding your model
Current API assumes that your model has:
 1. some parameters, that can be set before experiment.  
 2. one or several outputs that describe model state at the end of simulation interval. 
 3. one or several inputs that are action performed at each simulation step.
 
 An attempt to set value for a variable that is defined as constant in a model will cause crash.
 Thus, no of mentioned variables in a model should be constants. 
 
 For the cart pole FMU:
 1. these are staring conditions and model parameters:
    * theta_0 - angle of the pole, when experiment starts.
        It is counted from the positive direction of X-axis. Specified in radians.
        1/2*pi means pole standing straight on the cart.
    * theta_dot_0 - initial angle speed of a pole;
    * m_cart - mass of a cart;
    * m_pole - mass of a pole.
 2. State variables:
    * x - cart position;
    * x_dot - cart velocity;
    * theta - pole angle;
    * theta_dot - angle speed of pole.
 3. Action:
    * f - magnitude of the force applied to a cart at each time step.
 
## 2. Creating a class

### Inheritance hierarchy
To use your FMU for simulation of an environment in the Gym, 
you should create a class describing this environment. 

All abstract logic and default behaviour was already implemented in a toolbox.

To reuse it, inherit your class from `FMI2CSEnv` or `FMI1CSEnv`, 
depending on what FMI standard was used to compile an FMU.

```python
class JModelicaCSCartPoleEnv(FMI2CSEnv):
    ...
```

*Note*: in the `examples/cart_pole_env.py` environment class was also inherited from 
``CartPoleEnv`` class to avoid code duplication, during toolbox testing. 
This class contains all abstract logic of a cart-pole system.
To test the toolbox capability to work with FMU's exported in different tools (Dymola, Modelica), 
we used 2 FMU's compiled from the same model specification. 
In this case, two environment classes should be written, but code extraction allowed to write common logic just once.

At the same time, toolbox was tested on FMU compiled with different FMI standard versions: 1.0 and 2.0. 
This is shown in inheritance structure: while ``JModelicaCSCartPoleEnv`` inherits ``FMI2CSEnv``, ``DymolaCSCartPoleEnv``
inherits ``FMI1CSEnv``.

> **Disclaimer**: It is strongly advised to test environment behaviour to ensure relevant results, 
when a new FMU is integrated. Certain adjustments in implementation may be required.
Although FMI standard was developed to ensure tool-independent model sharing, 
during experiments with Cart-pole and other systems, authors have faced differences/particularities in behaviour
of FMUs compiled in different tools. E.g. automatic seeding of random generators works differently in Dymola and 
JModelica-compiled FMUs.
  
### Constructor and configuration
Next, you should define class constructor. It is advised to use all model parameters as class attributes,
so you can configure your environment during experiments.

Here we initialize some attributes of the environment, like:
 * thresholds for cart-pole position to determine if experiment has ended.
 * Force magnitude to be applied to the cart. Direction is chosen by action on each experiment step.
 * Attributes necessary for rendering, in particular, pole and cart transformations that are used to
  move corresponding objects on rendering.
 
 We also set logging level.
 
```python
    def __init__(self,
                 m_cart,
                 m_pole,
                 theta_0,
                 theta_dot_0,
                 time_step,
                 positive_reward,
                 negative_reward,
                 force,
                 log_level):

        logger.setLevel(log_level)

        self.force = force
        self.theta_threshold = TWELVE_DEGREES_IN_RAD
        self.x_threshold = 2.4

        self.viewer = None
        self.display = None
        self.pole_transform = None
        self.cart_transform = None
```

As parent class requires configuration of your model for execution of abstract logic, we will provide it.
Note, that we set values for configuration with parameters passed to the constructor.
Then, we call initialization method of a parent class.

```python
        config = {
            'model_input_names': 'f',
            'model_output_names': ['x', 'x_dot', 'theta', 'theta_dot'],
            'model_parameters': {'m_cart': m_cart, 'm_pole': m_pole,
                                 'theta_0': theta_0, 'theta_dot_0': theta_dot_0},
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward
        }
        super().__init__("../resources/jmodelica/linux/ModelicaGym_CartPole_CS.fmu",
                         config, log_level)
```

Configuration field have following meaning:
* ``model_input_names`` and ``model_output_names`` are names of model inputs and outputs
 that we clarified on previous step.
 *  ``model_parameters`` is dictionary of names of model parameters with the corresponding values.
 * ``time_step`` is a time interval between simulation steps in the environment.
 * ``positive_reward`` and ``negative_reward`` are positive and negative reward to be used 
 in default reward policy. We will discuss it on the next step.
 
## 3. Implementing required interfaces

As it was mentioned, some logic is implemented and ready to be used out of box. 
E.g. default reward policy: reward Reinforcement Learning agent for each step of experiment,
penalize, when experiment is done. This way, agent is encouraged to make experiment last as long as possible.
However, if you want to use more sofisticated rewarding strategy, just override ``_reward_policy`` method:
```python
def _reward_policy(self):
    ...
    return "reward depending on current environment state stored in self.state and self.done"
```

In the same manner you can override any logic you like. This makes our toolbox very flexible.
However, for usual use cases only custom logic in ``_is_done`` is required

Implementation of some other methods is required. These are: 
``_is_done``, ``_get_action_space``, ``_get_observation_space``, ``step`` and 
(if you are going to visualize your experiments) ``render``

First, we will implement internal logic:
```python
def _is_done(self):
    """
    Internal logic that is utilized by parent classes.
    Checks if cart position or pole angle are inside required bounds, defined by thresholds:
    x_threshold - 2.4
    angle threshold - 12 degrees

    :return: boolean flag if current state of the environment indicates that experiment has ended.
    True, if cart is not further than 2.4 from the starting point
    and angle of pole deflection from vertical is less than 12 degrees
    """
    x, x_dot, theta, theta_dot = self.state
    logger.debug("x: {0}, x_dot: {1}, theta: {2}, theta_dot: {3}".format(x, x_dot, theta, theta_dot))

    theta = abs(theta - NINETY_DEGREES_IN_RAD)

    if abs(x) > self.x_threshold:
        done = True
    elif theta > self.theta_threshold:
        done = True
    else:
        done = False

    return done
```
 
 This method determines if experiment has ended during the simulation step execution.
 Logic depends on problem formulation. For cart-pole problem stop conditions are cart moving too far from start
  and pole falling more than 12 degrees from vertical position.
  
  Next two methods are easy to implement. You just have to describe variable spaces of model inputs and outputs,
  using one or some classes from ``spaces`` package of OpenAI Gym library.
  As there are only two discrete actions (push left or push right) and continuous bounds on state space variables.
  implementation is as follows:
  
  ```python
    def _get_action_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns action space according to OpenAI Gym API requirements

        :return: Discrete action space of size 2, as only 2 actions are available: push left or push right.
        """
        return spaces.Discrete(2)

    def _get_observation_space(self):
        """
        Internal logic that is utilized by parent classes.
        Returns observation space according to OpenAI Gym API requirements

        :return: Box state space with specified lower and upper bounds for state variables.
        """
        high = np.array([self.x_threshold, np.inf, self.theta_threshold, np.inf])
        return spaces.Box(-high, high)

```  

Now, we should implement OpenAI Gym API methods. 

``step`` executes one step of an experiment - simulation in defined time interval. 
Common logic is abstract, so it was implemented in parent class. We just add transformation of action alias
 (push left if action value is < 0, and right - otherwise) into 
the exact value of force magnitude.

```python
    def step(self, action):
        """
        OpenAI Gym API. Executes one step in the environment:
        in the current state perform given action to move to the next action.
        Applies force of the defined magnitude in one of two directions, depending on the action parameter sign.

        :param action: alias of an action to be performed. If action > 0 - push to the right, else - push left.
        :return: next (resulting) state
        """
        action = self.force if action > 0 else -self.force
        return super().step(action)
```

To visualize our experiments we will use built-in Gym tools:
```python
    def render(self, mode='human', close=False):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        Draws cart-pole with the built-in gym tools.

        :param mode: rendering mode. Read more in Gym docs.
        :param close: flag if rendering procedure should be finished and resources cleaned.
        Used, when environment is closed.
        :return: rendering result
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return True

        screen_width = 600
        screen_height = 400

        scene_width = self.x_threshold * 2
        scale = screen_width / scene_width
        cart_y = 100  # TOP OF CART
        pole_width = 10.0
        pole_len = scale * 1.0
        cart_width = 50.0
        cart_height = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)

            # add cart to the rendering
            l, r, t, b = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.cart_transform = rendering.Transform()
            cart.add_attr(self.cart_transform)
            self.viewer.add_geom(cart)

            # add pole to the rendering
            pole_joint_depth = cart_height / 4
            l, r, t, b = -pole_width / 2, pole_width / 2, pole_len - pole_width / 2, -pole_width / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.pole_transform = rendering.Transform(translation=(0, pole_joint_depth))
            pole.add_attr(self.pole_transform)
            pole.add_attr(self.cart_transform)
            self.viewer.add_geom(pole)

            # add joint to the rendering
            joint = rendering.make_circle(pole_width / 2)
            joint.add_attr(self.pole_transform)
            joint.add_attr(self.cart_transform)
            joint.set_color(.5, .5, .8)
            self.viewer.add_geom(joint)

            # add bottom line to the rendering
            track = rendering.Line((0, cart_y - cart_height / 2), (screen_width, cart_y - cart_height / 2))
            track.set_color(0, 0, 0)
            self.viewer.add_geom(track)

        # set new position according to the environment current state
        x, _, theta, _ = self.state
        cart_x = x * scale + screen_width / 2.0  # MIDDLE OF CART

        self.cart_transform.set_translation(cart_x, cart_y)
        self.pole_transform.set_rotation(theta - NINETY_DEGREES_IN_RAD)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
```

Note, that this method also contains logic for proper ending of rendering process.
You are advised to do so as well.

Finally, we define, how to close the environment. In our case it just means end rendering process:

```python
    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return self.render(close=True)
```

Note: in `examples/cart_pole_env.py` file you will find ``DymolaCSCartPoleEnv`` and ``JModelicaCSCartPoleEnv``. 
These are wrappers and as it was explained above, these is caused by toolbox testing particularities.

## Thank you for reading this to the end. If you reached this point, you are ready to use custom environments, simulated with FMU's. 
