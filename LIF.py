import logging.config
import numpy as np
from Core import Core

# Logging configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


class LIF(Core):
    """"
    This is a class representation of the Leaky Integrate-and-Fire neuron model. The dynamics of the model are
    described by the equation:

    .. math::
        \tau_{m} \frac{dv}{dt} = -v(t)+RI(t)  |  \tau_{m}=RC

    where v is the membrane potential, R the resistance, I the injected current, C the capacitance and t the time.

    :param spike_threshold: The threshold voltage value for emitting a spike, defaults to 0.1
    :type spike_threshold: float, optional
    :param reset_potential: The value of the resting state of the neuron model after emitting a spike, defaults to 0
    :type reset_potential: float, optional
    :param absolute_refractory_period: The absolute refractory period when the neuron cannot process any input after
            emitting a spike, defaults to 4
    :type absolute_refractory_period:  int, optional
    :param resistance: The resistance value of the simple resistor-capacitor (RC) circuit that describes the neuron's
            membrane potential, defaults to 1.0
    :type resistance: float, optional
    :param capacitance: The capacitance value of the simple resistor-capacitor (RC) circuit that describes the neuron's
            membrane potential, defaults to 10.0
    :type capacitance: float, optional
    :param tau: The step size of the Euler method for solving first order differential equations, defaults to 1
    :type tau: float, optional

    Attributes
    ----------
    _refractory_time : int
        The refractory time while the neuron cannot process any input after emitting a spike
    """

    def __init__(self, spike_threshold: float = 0.1, reset_potential: float = 0, absolute_refractory_period: int = 4,
                 resistance: float = 1.0, capacitance: float = 10.0, tau: float = 1.0):
        """
        Constructor method
        """
        self._spike_threshold = spike_threshold
        self._reset_potential = reset_potential
        self._absolute_refractory_period = absolute_refractory_period
        self._resistance = resistance
        self._capacitance = capacitance
        self.tau = tau
        self._refractory_time = -1
        self.parameter_bounds = [0.01, 2.5], [0, 5]

    def compute_membrane_potential(self, time_point: int, current: float) -> float:
        """
        Calculates the membrane potential of the LIF model

        :param time_point: The time at which the neuron is stimulated with the specified current
        :param current: The injected current
        :return: The membrane potential produced by the injected current at the specified time point
        :rtype: float
        """
        self.fired = False
        if time_point > self._refractory_time:
            self.membrane_potential += self._tau * (-(self.membrane_potential - current * self._resistance) /
                                                    (self._resistance * self._capacitance))
            if self.membrane_potential > self._spike_threshold:
                self.fired = True
                self.membrane_potential = self._reset_potential
                self._refractory_time = time_point + self._absolute_refractory_period
                self.num_firings += 1
                self.firing_times.append(time_point)
        if time_point > 0:
            self.firing_rate = len(self.firing_times) / (time_point+1)  # Plus one because the time_point starts with 0

        return self.membrane_potential

    def get_firing_rate(self, *current):
        """
        Gets the firing rate of the neuron model. If no current is specified, the function will return the last
        firing rate computed in the last simulation. If the current is a scalar then the function will return the mean
        firing rate of the neuron

        .. math::
        f= \left [ \Delta_{abs} + \tau_{m} ln \frac{RI}{RI-v_{th}} \right ]^{-1}

        If the current is a vector (signal) then the function will stimulate the neuron for the length (duration) of the
        vector (signal).

        :param current: the injected current
        :return: the firing rate that the neuron model produces with the specified current
        """
        if len(current) == 1:
            current = current[0]
            if np.isscalar(current):
                # If it is a constant current
                tm = self._resistance * self._capacitance
                f = 1 / (self._absolute_refractory_period + (tm * np.log(
                    (self._resistance * current) / ((self._resistance * current) - self._spike_threshold))))
                return f
            else:
                stimulation_time = len(current)
                self.reset_core()
                for t in range(stimulation_time):
                    self.compute_membrane_potential(t, current[t])
                return self._firing_rate
        else:
            # If no current is specified then the function returns the firing rate of the last simulation
            return self._firing_rate

    def reset_core(self):
        """"
        Reset the functional attributes of the LIF model, i.e., fired = False, membrane_potential = reset_potential,
        firing_times.clear(), firing_rate = 0, _refractory_time = -1, nmm_firings = 0.
        """
        logger.debug("Resetting the LIF attributes")
        # Reset the intrinsic functional properties
        self.fired = False
        self.membrane_potential = self._reset_potential
        self.firing_times.clear()
        self.firing_rate = 0
        # Reset the specific functional  properties of the LIF model
        self._refractory_time = -1
        self.num_firings = 0

    def set_parameters(self, parameters: list[float]):
        """
        Set the voltage threshold and the absolute refractory period parameters of the LIF. These parameters can be
        updated during an optimisation process.

        :param parameters: The list of parameter values
        """
        if len(parameters) == 2:
            self._spike_threshold = parameters[0]
            self._absolute_refractory_period = int(parameters[1])
        return self

    def get_parameters(self) -> list[float]:
        """
        Returns a list with the voltage threshold and the absolute refractory period parameters of the LIF.

        :return: The list of parameter values
        """
        parameters = [self._spike_threshold, self._absolute_refractory_period]
        return parameters

    def get_parameters_bounds(self) -> tuple[[float, float], ...]:
        """
        Returns a tuple containing the bounds of the threshold and refractory time.

         :return: The bounds of the LIF parameters
        """
        return self.parameter_bounds

    def get_num_firings(self) -> int:
        """
        Return the number of spikes that the neuron model emitted after being stimulated with an injected current.

        :return: The number of spikes
        """
        return self.num_firings

    def get_spike_threshold(self) -> float:
        """
        Returns the threshold value of the membrane potential for emitting a spike.

        :return: The threshold value
        """
        return self._spike_threshold

    def set_spike_threshold(self, spike_threshold: float):
        """
        Sets the threshold value of the membrane potential for emitting a spike.

        :param spike_threshold: The spike threshold
        """
        self._spike_threshold = spike_threshold

    def get_reset_potential(self) -> float:
        """
        Returns the reset potential, i.e., the value of the resting state of the neuron model after emitting a spike.

        :return: The reset potential
        """
        return self._reset_potential

    def set_reset_potential(self, reset_potential: float):
        """
        Sets the reset potential, i.e., the value of the resting state of the neuron model after emitting a spike.

        :param reset_potential: The reset potential
        """
        self._reset_potential = reset_potential

    def get_absolute_refractory_period(self) -> int:
        """
        Returns the value of the neuron's absolute refractory period.

        :return: The absolute refractory period
        """
        return self._absolute_refractory_period

    def set_absolute_refractory_period(self, absolute_refractory_period: int):
        """
        Sets the value of the neuron's absolute refractory period. If the parameter is a float value, the function
        converts it into an int value

        :param absolute_refractory_period: The absolute refractory period
        """
        self._absolute_refractory_period = int(absolute_refractory_period)

    def get_resistance(self) -> float:
        """
        Gets the resistance value of the simple resistor-capacitor (RC) circuit that describes the dynamics of the
        neuron's membrane potential.

        :return: The resistance of the RC circuit
        """
        return self._resistance

    def set_resistance(self, resistance: float):
        """
        Sets the resistance value of the simple resistor-capacitor (RC) circuit that describes the dynamics of the
        neuron's membrane potential.

        :param resistance: The resistance of the RC circuit
        """
        self._resistance = resistance

    def get_capacitance(self) -> float:
        """
        Gets the capacitance value of the simple resistor-capacitor (RC) circuit that describes the dynamics of the
        neuron's membrane potential.

        :return: The capacitance of the RC circuit
        """
        return self._capacitance

    def set_capacitance(self, capacitance: float):
        """
        Sets the capacitance value of the simple resistor-capacitor (RC) circuit that describes the dynamics of the
        neuron's membrane potential.

        :param capacitance: The capacitance of the RC circuit
        """
        self._capacitance = capacitance

    def get_name(self) -> str:
        """
        Returns the name of the neuron model

        :return: The name of the LIF model
        """
        return "Leaky Integrate and Fire"

    def clone(self) :
        """
        Creates a copy of the neuron model.

        :return: A copy of the neuron model
        :rtype: LIF object
        """
        return LIF(spike_threshold=self._spike_threshold, reset_potential=self._reset_potential,
                   absolute_refractory_period=self._absolute_refractory_period)
