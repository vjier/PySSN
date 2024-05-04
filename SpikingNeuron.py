from Core import Core
from enum import Enum
import numpy as np


class Type(Enum):
    EXCITATORY = 1
    INHIBITORY = -1
    NEUTRAL = 0

    def describe(self):
        return self.name, self.value


class SpikingNeuron:
    _input_synapses = []
    _output_synapses = []
    _core = Core
    type = Type.EXCITATORY

    def run(self, current):
        simulation_time = len(current)
        self._core.reset_core()
        for t in range(simulation_time):
            self._core.compute_membrane_potential(t, current[t])

    def clear(self):
        self._core.reset_core()

    def get_core(self):
        return self._core

    def set_core(self, core):
        self._core = core

    def create_input_synapses(self, num_synapses):
        self._input_synapses = np.zeros(num_synapses)

    def set_input_synapses(self, input_synapses):
        self._input_synapses = np.array(input_synapses)

    def get_input_synapses(self):
        return self._input_synapses

    def get_firing_times(self):
        return self._core.get_firing_times()
