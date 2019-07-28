from modelicagym.environment import ModelicaBaseEnv, ModelicaType
import logging

logger = logging.getLogger(__name__)


class ModelicaCSEnv(ModelicaBaseEnv):
    """
    Wrapper class of ModelicaBaseEnv for convenient creation of environments that utilize
    FMU exported in co-simulation mode.
    Should be used as a superclass for all such environments.
    Implements abstract logic, handles different Modelica types (JModelica, Dymola) particularities.

    """

    def __init__(self, model_path, config, type, log_level):
        """

        :param model_path: path to the model FMU. Absolute path is advised.
        :param config: dictionary with model specifications. For more details see ModelicaBaseEnv docs
        :param type: kind of the Modelica tool that was used for FMU compilation.
        :param log_level: level of logging to be used in experiments on environment.
        """
        self.type = type
        logger.setLevel(log_level)
        super().__init__(model_path, config, log_level)

    def reset(self):
        """
        OpenAI Gym API. Restarts environment and sets it ready for experiments.
        In particular, does the following:
            * resets model
            * sets simulation start time to 0
            * sets initial parameters of the model
            * initializes the model
            * sets environment class attributes, e.g. start and stop time.
        :return: state of the environment after resetting
        """
        logger.debug("Experiment reset was called. Resetting the model.")

        self.model.reset()
        # particularity of FMU exported from JModelica
        if self.type == ModelicaType.JModelica:
            self.model.setup_experiment(start_time=0)

        self._set_init_parameter()
        self.model.initialize()

        # get initial state of the model from the fmu
        self.start = 0
        self.stop = 0
        self.state = self.do_simulation()

        self.stop = self.tau
        self.done = self._is_done()
        return self.state


class DymolaCSEnv(ModelicaCSEnv):
    """
    Wrapper class.
    Should be used as a superclass for all environments using FMU exported from Dymola in co-simulation mode.
    Abstract logic is implemented in parent classes.

    Refer to the ModelicaBaseEnv docs for detailed instructions on own environment implementation.
    """

    def __init__(self, model_path, config, log_level):
        super().__init__(model_path, config, ModelicaType.Dymola, log_level)


class JModCSEnv(ModelicaCSEnv):
    """
    Wrapper class.
    Should be used as a superclass for all environments using FMU exported from JModelica in co-simulation mode.
    Abstract logic is implemented in parent classes.

    Refer to the ModelicaBaseEnv docs for detailed instructions on own environment implementation.
    """
    def __init__(self, model_path, config, log_level):
        super().__init__(model_path, config, ModelicaType.JModelica, log_level)
