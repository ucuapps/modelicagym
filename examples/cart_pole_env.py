"""
Classic cart-pole example implemented with an FMU simulating a cart-pole system.
Implementation inspired by OpenAI Gym examples:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

import logging
import math
import numpy as np
from gym import spaces
from modelicagym.environment import JModCSEnv, DymolaCSEnv

logger = logging.getLogger(__name__)


NINETY_DEGREES_IN_RAD = (90 / 180) * math.pi
TWELVE_DEGREES_IN_RAD = (12 / 180) * math.pi


class CartPoleEnv:
    """
    Class extracting common logic for JModelica and Dymola environments for CartPole experiments.
    Allows to avoid code duplication.
    Implements all methods for connection to the OpenAI Gym as an environment.


    """


    # modelicagym API implementation
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

    # OpenAI Gym API implementation
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

    # This function was heavily inspired by OpenAI example:
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
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

    def close(self):
        """
        OpenAI Gym API. Closes environment and all related resources.
        Closes rendering.
        :return: True if everything worked out.
        """
        return self.render(close=True)


class JModelicaCSCartPoleEnv(CartPoleEnv, JModCSEnv):
    """
    Wrapper class for creation of cart-pole environment using JModelica-compiled FMU.

    Attributes:
        m_cart (float): mass of a cart.

        m_pole (float): mass of a pole.

        theta_0 (float): angle of the pole, when experiment starts.
        It is counted from the positive direction of X-axis. Specified in radians.
        1/2*pi means pole standing straight on the cast.

        theta_dot_0 (float): angle speed of the poles mass center. I.e. how fast pole angle is changing.
        time_step (float): time difference between simulation steps.
        positive_reward (int): positive reward for RL agent.
        negative_reward (int): negative reward for RL agent.
    """

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

        config = {
            'model_input_names': 'f',
            'model_output_names': ['x', 'x_dot', 'theta', 'theta_dot'],
            'model_parameters': {'m_cart': m_cart, 'm_pole': m_pole,
                                 'theta_0': theta_0, 'theta_dot_0': theta_dot_0},
            'initial_state': (0, 0, 85 / 180 * math.pi, 0),
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward
        }
        super().__init__("../resources/jmodelica/linux/ModelicaGym_CartPole_CS.fmu",
                         config, log_level)


class DymolaCSCartPoleEnv(CartPoleEnv, DymolaCSEnv):
    """
    Wrapper class for creation of cart-pole environment using Dymola-compiled FMU.

    Attributes:
        m_cart (float): mass of a cart.

        m_pole (float): mass of a pole.

        phi1_start (float): angle of the pole, when experiment starts.
        It is counted from the positive direction of X-axis. Specified in radians.
        1/2*pi means pole standing straight on the cast.

        w1_start (float): angle speed of the poles mass center. I.e. how fast pole angle is changing.
        time_step (float): time difference between simulation steps.
        positive_reward (int): positive reward for RL agent.
        negative_reward (int): negative reward for RL agent.
    """

    def __init__(self,
                 m_cart,
                 m_pole,
                 phi1_start,
                 w1_start,
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

        config = {
            'model_input_names': 'u',
            'model_output_names': ['s', 'v', 'phi1', 'w'],
            'model_parameters': {'m_trolley': m_cart, 'm_load': m_pole,
                                 'phi1_start': phi1_start, 'w1_start': w1_start},
            'initial_state': (0, 0, 85 / 180 * math.pi, 0),
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward
        }
        # loads FMU corresponding to the Modelica type required
        super().__init__("../resources/dymola/linux/ModelicaGym_CartPole.fmu",
                         config, log_level)

