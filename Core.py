from abc import ABC, abstractmethod

import numpy as np


class Core(ABC):
    """
    A base class for implementing spiking neuron models.

        Attributes
        ----------
         _membrane_potential : float
            Describes the current membrane potential of the neuron model. The value is calculated in the :meth:
            `compute_membrane_potential`.
        _fired : bool,
            Describes the last firing status of the neuron model. The status is updated in the :meth:
            `compute_membrane_potential`.
        _firing_rate : float
            Describes the firing rate of the neron after being stimulated with a current.
        _num_firings : float
            The number of spikes that the neuron produces after being stimulated with a current
        _firing_times :list[int]
            An array that contains the firing times of the neuron model after a simulation. New elements are added when
            the neuron emits a spike while calculating the membrane potential in the :meth:`compute_membrane_potential`.
        _parameter_bounds : tuple
            A tuple containing the bounds of the model's parameters.
        _tau : int
            The step size for solving the differential equation(s)
    """

    _membrane_potential: float = 0
    _fired: bool = False
    _firing_rate: float = 0
    _num_firings: int = 0
    _firing_times: list = []
    _parameter_bounds: tuple = ()
    _tau: int = 1

    @abstractmethod
    def compute_membrane_potential(self, time_point: int, current: float) -> float:
        """"
        Calculates the membrane potential of a neuron model.

        :param time_point: The current simulation time point
        :param current: The injected current at the specified time point
        :return: The current membrane potential
        """
        pass

    @property
    def firing_rate(self) -> float:
        """
        Returns the firing rate that a neuron model produces after being stimulated with a current.

        :return: The firing rate
        """
        return self._firing_rate

    @firing_rate.setter
    def firing_rate(self, firing_rate: float):
        """"
        Sets the firing rate of the neuron model after being stimulated with an injected current.

        :param firing_rate: The firing rate
        """
        self._firing_rate = firing_rate

    @abstractmethod
    def get_firing_rate(self, *args) -> float:
        """
        Returns the firing rate that a neuron model produces after being stimulated with a current. The function
        of a child class must implement the cases where no current is specified or when the current is a scalar value
        (constant current) or a vector (time-varying input current).

        :param args: The current firing rate of the neuron model
        :type args: float, list[floats], optional
        :return: The firing rate
        """
        pass

    @abstractmethod
    def reset_core(self):
        """"
        Set the functional attributes of the neuron model to their initial condition before a simulation.
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: list[float]):
        """"
        Set the parameters of the neuron model. The number of parameters depend on the type of the neuron model.

        :param parameters: The list of the neuron's parameter values
        """
        pass

    @abstractmethod
    def get_parameters(self) -> list[float]:
        """"
        Returns the parameters of the neuron model. The number of parameters depend on the type of the neuron model.

        :return: A list with the values of the parameters of the neuron model
        """
        pass

    @abstractmethod
    def get_parameters_bounds(self) -> tuple[[float, float], ...]:
        """
        Returns the parameters' bounds of the neuron model. The parameters and bounds depend on the neuron model. For
        example, the Leaky Integrate-and-Fire model has two parameter to optimise: 1) the threshold for emitting a spike
        and 2) the refractory time, which bounds could be defined between the ranges th=[0.01, 1.0] and rt=[1, 10]

        :return: A tuple containing the bounds of the model's parameters.
        """
        return self._parameter_bounds
        pass

    @property
    def parameter_bounds(self) -> tuple[[float, float], ...]:
        """
        Returns the parameters' bounds of the neuron model. The parameters and bounds depend on the neuron model. For
        example, the Leaky Integrate-and-Fire model has two parameter to optimise: 1) the threshold for emitting a spike
        and 2) the refractory time, which bounds could be defined between the ranges th=[0.01, 1.0] and rt=[1, 10]

        :return: A tuple containing the bounds of the model's parameters.
        """
        return self._parameter_bounds

    @parameter_bounds.setter
    def parameter_bounds(self, parameter_bounds: tuple[[float, float], ...]):
        """
        Sets the parameter bounds of the neuron model.

        :param parameter_bounds: The parameter bounds.
        """
        self._parameter_bounds = parameter_bounds

    @abstractmethod
    def get_name(self) -> str:
        """"
        Returns  the name of the neuron model.

        :return: The name of the neuron model
        """
        pass

    @abstractmethod
    def clone(self):
        """"Creates a new object of the type of the neuron model and copies the configuration attributes of the original
        object to the new object.

        :return: A copy of the neuron model
        :rtype: Core object
        """
        pass

    @property
    def membrane_potential(self) -> float:
        """"Returns the current membrane potential of the neuron model.

        :return: The current membrane potential
        :rtype: float
        """
        return self._membrane_potential

    @membrane_potential.setter
    def membrane_potential(self, membrane_potential: float):
        """"Set the current membrane potential.

        :param membrane_potential: The membrane potential of the neuron model
        :type membrane_potential: float
        """
        self._membrane_potential = membrane_potential

    @property
    def fired(self) -> bool:
        """"Returns the firing status of the model.

        :return: The firing status of the neuron model
        :rtype: bool
        """
        return self._fired

    @fired.setter
    def fired(self, status: bool):
        """"Sets the firing status of the model. The value is updated in the :meth:`compute_membrane_potential`.
        :param status: The firing status of the neuron model
        :type status: bool
        """
        self._fired = status

    @property
    def num_firings(self) -> int:
        """"Returns the number of spikes that the neuron model produced after being stimulated with a current. The value
        is updated in the :meth:`compute_membrane_potential`.
        :return: The number of spikes
        :rtype: int
        """
        return self._num_firings

    @num_firings.setter
    def num_firings(self, num_firings):
        """"Sets the number of firings.

        :param num_firings: The number of firings
        :type num_firings: int
        """
        self._num_firings = num_firings

    @property
    def firing_times(self) -> list[int]:
        """"Returns the times when the neuron model emitted a spike after being stimulated with an injected current .

        :return: A list of integers defining the firing times
        :rtype: list[int]
        """
        return self._firing_times

    @property
    def tau(self) -> float:
        """Returns the step size of the Euler method for solving the first order differential equations that defines the
        dynamics of the membrane potential of the neuron model.

        :return: The step size for solving the differential equation(s)
        """
        return self._tau

    @tau.setter
    def tau(self, tau: float):
        """Sets the step size of the Euler method for solving the first order differential equations that defines the
        dynamics of the membrane potential of the neuron model.

        :param tau: The step size for solving the differential equation(s)
        """
        self._tau = tau

    def get_firing_times(self) -> list[int]:
        """"Returns the times when the neuron model emitted a spike after being stimulated with an injected current .

        :return: A list of integers defining the firing times
        """
        return self._firing_times

    def get_spike_train(self, *current) -> np.ndarray:
        """
        Gets the spike train produced by an injected current. The spike train is the same size od the current.

        :param current: the injected current
        :return: a spike train
        """
        stimulation_time = len(current)
        spike_train = np.zeros(stimulation_time)
        self.reset_core()
        for t in range(stimulation_time):
            self.compute_membrane_potential(t, current[t])
            if self.fired:
                spike_train[t] = 1
        return spike_train
