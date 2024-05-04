# Author: Josafath Israel Espinosa Ramos <vjier1979@gmail.com>,

import logging.config
import pandas
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from Core import Core
from LIF import LIF
import json

import Gamma as gamma

# Logging configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


class SSNClassifier(BaseEstimator, ClassifierMixin):
    """
    A temporal data classifier which implements a single spiking neuron.

        Parameters
        ----------
        num_inputs: int
            The number of inputs that the neuron model will process
        core : core, default=LIF()
            The spiking neuron model used for classification
        neuron_param : array-like, default=[]
            A vector that contains the core's parameters
        weights : array-like, shape (1, n_weights), default=[]
            A vector that contains the weights of the neuron
        weight_bounds : array-like, shape (1,2), default=[-0.01,0.01]
            A vector that contains the lower and upper limits of the weights. This parameter is crucial for achieving
            good performance. Try to modify the default values in powers of 2, e.g., [-0.001, 0.001], [-0.01, 0.01],
            [-0.1, 0.1], [-1.0, 1.0].
        firing_rates : dict
            A dictionary that contains the average firing rates for each class. This attribute is overridden when
            calling the :meth:`fit` and the values depends on the data.
        firing_times :  dict
            A dictionary that contains the firing times that the samples belonging to each class produce. This attribute
            is used for classification of the data based on the gamma factor. This attribute is overridden when calling
            the :meth:`fit` and the values depends on the data.
        delta_window : float, default = 0.002 (time steps)
            The time interval for finding a coincidence between two spike times. These number can be optimised.
        base_model : dict, default = None
            The dictionary that contains the parameters of the snn classifier. dict:{'neuron_param':array,
            'weights':array, 'firing_rates':dict {int: float} (class_label, firing_rates)}

        Attributes
        ----------
        classes_ : ndarray, shape (n_classes,)
            The classes seen at :meth:`fit`.
       """

    def __init__(self, num_inputs: int =0, core: Core = LIF(), neuron_param: list = None, weights: list = None,
                 weight_bounds: list = None, firing_rates: dict = None, firing_times: dict = None,
                 delta_window: float = 0.001, base_model: dict = None):
        if neuron_param is None:
            neuron_param = []
        if weights is None:
            weights = []
        if weight_bounds is None:
            weight_bounds = [-1.0, 1.0]  #[-0.01, 0.01]
        if firing_rates is None:
            firing_rates = {}
        self.core = core
        self.num_inputs = num_inputs
        self.neuron_param = neuron_param
        self.weights = weights
        self.weight_bounds = weight_bounds
        self.firing_rates_ = firing_rates
        self.classes_ = {}
        self.firing_times_ = firing_times
        self.delta_window = delta_window
        if base_model is not None:
            self.set_params(**base_model)


    def fit(self, samples: list[pandas.DataFrame], targets: np.ndarray) -> object:
        """
        Implementation of the fitting function. Calculates the firing rates

        :param samples: The list of temporal samples (list of 2D array)
        :param targets: The list of target values
        :return: self
        """

        logger.debug("Fitting the SSN classification model using %s samples ", str(len(samples)))
        if len(samples) != len(targets):
            raise ValueError("The number of samples must be equal to the number of targets")
        self.core.set_parameters(self.neuron_param)
        # Store the classes seen during fit
        (unique, counts) = np.unique(targets, return_counts=True)
        self.classes_ = unique
        firing_rates = {}
        spike_trains = {}

        for i in range(len(samples)):
            # Get the firing rates of all samples and sum them for each class
            firing_rate = self._get_firing_rate(samples[i])
            prev = firing_rates.get(targets[i], 0)
            firing_rates[targets[i]] = prev + firing_rate

            # Get the firing times of all samples (for spike trains)
            # firing_times = self.core.get_firing_times()
            # spike_train = spike_trains.get(targets[i], np.zeros(len(samples[1].index)))
            # spike_train[firing_times] += 1
            # spike_trains[targets[i]] = spike_train

        # Calculate the average firing rate of each class and
        for i in range(len(unique)):
            firing_rates[unique[i]] = firing_rates.get(unique[i]) / counts[i]
            # spike_train = spike_trains.get(unique[i])
            # spike_train[spike_train < (counts[i]/2)] = 0
            # spike_train[spike_train >= (counts[i]/2)] = 1
            # spike_trains[unique[i]] = np.where(spike_train == 1)

        self.firing_rates_ = firing_rates
        #self.firing_times_ = spike_trains
        self.firing_times_ = {}
        return self

    def predict(self, samples: list[pandas.DataFrame]) -> np.ndarray:
        """
        Implementation of prediction for the single neuron classifier. This function verifies if the average firing
        rates exist.

        :param samples: The list of temporal samples
        :return: The label for each sample is the label of the closest firing rate seen during fit
        """
        logger.debug("Predicting %s samples", str(len(samples)))
        # Check if fit had been called
        check_is_fitted(self, ['firing_rates_'])

        predicted = np.zeros(len(samples))
        predicted2 = np.zeros(len(samples))
        classes_array = np.array(list(self.firing_rates_.items()))

        for i in range(len(samples)):
            firing_rate = self._get_firing_rate(samples[i])
            idx = np.argmin(abs(classes_array[:, 1] - firing_rate))
            predicted[i] = classes_array[idx, 0]

            # # spike train similarity
            # predicted_firing_times = self.core.get_firing_times()
            # factors = np.zeros(len(classes_array))
            # for j in range(len(classes_array)):
            #     target = self.firing_times_.get(classes_array[j][0])
            #     target = target[0]
            #     #target = np.asarray(target)
            #     factors[j] = gamma.get_factor(target, predicted_firing_times, self.delta_window)
            # idx = np.argmax(factors)
            # predicted2[i] = classes_array[idx, 0]

        return predicted

    def evaluate(self, samples: list[pandas.DataFrame], targets: np.ndarray, scoring=None) -> object:
        """
        Fits the model and creates new firing rates. Then, make predictions using the whole dataset too. This method is
        not recommended. Different datasets must be used for creating new firing rates (training dataset) and making
        predictions (test dataset).

        :param samples: the sample to evaluate
        :param targets: the target labels of the samples
        :param scoring: the performance metric
        :return: the performance value
        """
        self.fit(samples=samples, targets=targets)
        predicted_values = self.predict(samples)
        return scoring(targets, predicted_values)

    def get_params(self, deep=True) -> dict[str, object]:
        """
        Returns the parameters of the classifier.

        :param deep: Determines whether the method should return the parameters of sub-estimators (this can be ignored).
        :return: A dictionary of the __init__ parameters of the classifier, together with their values.
        """
        return {"core": self.core, "neuron_param": self.neuron_param, "weights": self.weights,
                "num_inputs": self.num_inputs}

    def set_params(self, **params) -> object:
        """
        Set the parameters of the classifier using a dictionary.

        :param params: The dictionary that maps the __init__ parameters (neuron_param, weights and firing_rates_) of the
            SSN classifier
        :return: self
        """
        for parameter, value in params.items():
            setattr(self, parameter, value)
        self.core.set_parameters(self.neuron_param)
        return self

    def set_array_params(self, params: np.ndarray):
        """
        Set the parameters of the classifier using an array. The first parameters belongs to the number of inputs and
        the remaining to the core's properties.

        :param params: The array that has the weights and the neuron's parameters of the SSN classifier
        :return: self
        """
        num_neuron_params = len(self.core.get_parameters())
        if len(params) < (self.num_inputs + num_neuron_params):
            raise ValueError("The length of the parameters is less than the dimension of the problem")
        self.weights = params[0:self.num_inputs]
        self.core.set_parameters(params[self.num_inputs: self.num_inputs + num_neuron_params])

    def _get_firing_rate(self, sample: pandas.DataFrame) -> float:
        """
        Returns the firing rate that the neuron produces after being stimulated with the temporal data sample.

        :param sample: The temporal data sample
        :return: The firing rate.
        """
        if len(self.weights) != sample.shape[1]:
            raise ValueError("The number of temporal variables must be equal to the number of weights")
        current = np.dot(sample, self.weights)
        firing_rate = self.core.get_firing_rate(current)
        return firing_rate

    def get_firing_rates(self) -> dict[int, float]:
        """
        Return the average firing rate that the neuron produces after being stimulated with all the temporal data
        samples belonging to each class.

        :return: A dictionary containing the class labels along with the average firing rate
        """
        return self.firing_rates_

    def get_firing_times(self, sample: pandas.DataFrame) -> list[int]:
        """"
        Return the firing times that the neuron produces after being stimulated with the temporal data sample.

        :param sample: Temporal data sample
        :return: A list of the firing times
        """
        self.core.reset_core()
        current = np.dot(sample, self.weights)
        for t in range(len(current)):
            self.core.compute_membrane_potential(t, current[t])
        firing_times = self.core.get_firing_times()
        return firing_times

    def _get_spike_train(self, sample: pandas.DataFrame) -> np.ndarray:
        """
        Return the spike train that the neuron produces after being stimulated with the temporal data sample.

        :param sample: Temporal data sample
        :return: a spike train
        """
        if len(self.weights) != sample.shape[1]:
            raise ValueError("The number of temporal variables must be equal to the number of weights")
        current = np.dot(sample, self.weights)
        spike_train = self.core.get_spike_train(current)
        return spike_train

    def get_parameters_bounds(self) -> tuple:
        """
        Returns the bounds of the weights and the neuron properties to be optimised.

        :return: The bounds of the weights and the neuron properties to be optimised
        """
        bounds = tuple(self.weight_bounds for _ in range(self.num_inputs))
        bounds = bounds + self.core.get_parameters_bounds()
        return bounds

    def get_attributes(self) -> dict[str, object]:
        """
        Returns a dictionary mapping the neuron_param, weights and firing_rates_ attributes.

        :return: A dictionary mapping the attributes that defines a SSN classifier model
        """
        att = {"neuron_param": self.core.get_parameters(), "weights": self.weights, "firing_rates_": sorted(
            self.firing_rates_.items())}
        return att

    def save_to_json(self, file_name: str = ""):
        model_dict = {
            "neuron_param": self.core.get_parameters(),
            "weights": self.weights,
            "firing_rates_": sorted(self.firing_rates_.items()),
            "num_inputs:": self.num_inputs
        }
        with open(file_name, 'w') as outfile:
            json.dump(model_dict, outfile)

    def read_from_json(self, file_name: str = ""):
        with open(file_name) as json_file:
            model_dict = json.load(json_file)
            self.set_params(**model_dict)

