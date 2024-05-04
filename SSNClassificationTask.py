import logging.config
import pandas
import numpy as np
import numbers
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from scipy.optimize import OptimizeResult
from sklearn.metrics import accuracy_score
from scipy.optimize import differential_evolution
from SSNClassifier import SSNClassifier
from Core import Core
from LIF import LIF

from sklearn.model_selection import GridSearchCV
# Logging configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


def check_cv(cv):
    cv = 5 if cv is None else cv

    if isinstance(cv, numbers.Integral):
        return StratifiedKFold(cv)
    else:
        return cv


class SNNClassificationTask:
    """
    A classification task based on a Single Spiking Neuron (SSN) classifier.

        Parameters
        ----------
        num_inputs: int
            The number of inputs that the neuron model will process
        core : core
            The spiking neuron model used for classification
        max_iterations : int
            Maximum number of iterations of the differential evolution
        display : bool
            Indicates if the differential evolution prints the evaluated func at every iteration
        cv_outer : StratifiedKFold
            The number of folds of the outer stratified K-fold cross-validation (nested cross-validation)
        cv_inner : int
            The number of folds of the inner stratified K-fold cross-validation
        Attributes
        ----------
        _ssn_classifier: SSNClassifier
            The single spiking neuron classifier

   """

    def __init__(self, snn_classifier: SSNClassifier = None, num_inputs: int = 0, core: Core = LIF(),
                 max_iterations: int = 100, de_display: bool = True, cv_outer: StratifiedKFold = None,
                 cv_inner: StratifiedKFold = None, initial_population=None):
        if snn_classifier is None:
            snn_classifier = SSNClassifier(num_inputs, core=core)
        self._num_inputs = num_inputs
        self._core = core
        self._max_iterations = max_iterations
        self._display = de_display
        self._cv_outer = cv_outer
        self._cv_inner = cv_inner
        self._initial_population = initial_population
        self._ssn_classifier = snn_classifier

    @property
    def ssn_classifier(self):
        return self._ssn_classifier

    @ssn_classifier.setter
    def ssn_classifier(self, ssn_classifier: SSNClassifier):
        self._ssn_classifier = ssn_classifier

    def get_ssn_classifier(self) -> SSNClassifier:
        """
        Returns the SSN classifier.

        :return: The SSN classifier
        """
        return self._ssn_classifier

    def nested_cross_validation(self, data: list[pandas.DataFrame], targets: np.ndarray, cv_outer:
                                StratifiedKFold = None, cv_inner: StratifiedKFold = None) -> dict:
        """
        Nested cross-validation for Differential Evolution-Spiking Neuron classifier a classification dataset.

        :param data: The training dataset
        :param targets: The targets of the training dataset
        :param cv_outer: Determines the outer cross-validation splitting strategy. Possible inputs for cv are:
                        None, to use the default 5-fold cross validation,
                        int, to specify the number of folds in a (Stratified) KFold.        
        :param cv_inner: Determines the inner cross-validation splitting strategy. Possible inputs for cv are:
                        None, to use the default 5-fold cross validation,
                        int, to specify the number of folds in a (Stratified) KFold.      
        :return: A dictionary with the results of the nested cross-validation. dict={'scores': list, 'mean_acc': float,
         'std_acc': float, 'best_fitness': list, 'best_parameters': list, 'best_iter_fitness': list, 
         'best_firing_rates': list(dict)}
        """

        scores = list()
        best_fitness = list()
        best_parameters = list()
        best_iter_fitness = list()
        best_firing_rates = list()
        # Set the outer cross-validation procedure
        self._cv_outer = check_cv(cv_outer)
        # Set the inner cross-validation procedure
        self._cv_inner = check_cv(cv_inner)

        fold_id = 1
        logger.info("Nested cross_validation (%s folds)", str(self._cv_outer.n_splits))
        for training_index, testing_index in self._cv_outer.split(data, targets):
            logger.info("Outer fold %s / %s", str(fold_id), str(self._cv_outer.n_splits))
            # split data
            training_data = [data[i] for i in training_index]
            testing_data = [data[i] for i in testing_index]
            training_targets, testing_targets = np.array(targets[training_index]), np.array(targets[testing_index])
            # Execute the Differential Evolution. The inner cross-validation is embedded in the objective function
            result = self.tune(training_data, training_targets)
            # Set the weights and neuron parameters
            self.ssn_classifier.set_array_params(result.x)
            # Fit the model one more time on the whole training set
            model = self.fit(training_data, training_targets)
            # Evaluate the model on the test (hold out) dataset
            acc = model.score(testing_data, testing_targets)
            # Store the results
            scores.append(acc)
            best_fitness.append(result.fun)     # optimisation result, best agent fitness (training accuracy)
            best_parameters.append(result.x)    # best agent
            best_iter_fitness.append(result.best_agent_values)
            best_firing_rates.append(model.get_firing_rates())
            # report progress
            logger.info("Fold %s > acc=%.3f, best_fitness=%.3f ", fold_id, acc, result.fun)
            #print(testing_index)
            fold_id += 1

        # summarize the estimated performance of the model
        outer_results = {"scores": scores, "mean_acc": np.mean(scores), "std_acc": np.std(scores),
                         "best_fitness": best_fitness, "best_parameters": best_parameters,
                         "best_iter_fitness": best_iter_fitness, "best_firing_rates": best_firing_rates}
        return outer_results

    def tune(self, data: list[pandas.DataFrame], targets: np.ndarray) -> OptimizeResult:
        """
        Tunes the model's hyperparameters (weights and neuron parameters) and returns a dictionary of type
        OptimizeResult (see differential_evolution and OptimizeResult from scipy.optimize) with an additional item
        "best_agent_values", which is a list of floats describing the best agent fitness of each iteration of the
        differential evolution.

        :param data: The training dataset
        :param targets: The targets of the training dataset
        :return: The optimization result represented as a OptimizeResult object. Useful attributes are x: the solution
        of the optimisation; fun: the value of the objective function; and best_agent_values: the value of the best
        agent fitnes of each iteration of the differential evolution.
        """
        logger.info("Tuning the SSN classifier using %s samples", str(len(data)))
        if len(data) != len(targets):
            raise ValueError("The number of samples must be equal to the number of targets")
        self.ssn_classifier.core = self._core
        bounds = self.ssn_classifier.get_parameters_bounds()
        result = self._optimise_model(data, targets, bounds, init=self._initial_population)

        return result

    def _optimise_model(self, data: list[pandas.DataFrame], targets: np.ndarray, bounds: tuple[[float, float], ...],
                        init="latinhypercube") -> OptimizeResult:
        """
        Runs the differential evolution and returns the best parameters of the SSN classifier, i.e., weights, and the
        parameters of the spiking neuron. The differential evolution library scipy.optimize differential_evolution
        located in the _differentialevolution.py file was modified for recording the fitness values of the best agents
        in each iteration. Therefore, the function returns a OptimizeResult dictionary (see differential_evolution and
        OptimizeResult from scipy.optimize) with an additional item "best_agent_values", which is a list of floats
        describing the best agent fitness of each iteration of the differential evolution.

        :param data: The training dataset
        :param targets: The targets of the training dataset
        :param bounds: The bounds of the neuron's parameters
        :param init: Specify which type of population initialization is performed
        :return: A dictionary with the results of the differential evolution
        """
        logger.info("Running differential evolution")
        if init is None:
            init = "latinhypercube"

        pop = []
        energies = []

        result = differential_evolution(func=self._objective_function, bounds=bounds, args=(data, targets),
                                        maxiter=self._max_iterations, strategy='rand2bin', mutation=0.1,
                                        recombination=0.7, disp=self._display, popsize=10, updating='deferred',
                                        workers=-1, atol=0.0, tol=0.0, seed=None, init=init)

        logger.debug(result)
        logger.info("Differential evolution completed")
        return result

    def _objective_function(self, parameters: np.ndarray, data: list[pandas.DataFrame], targets: np.ndarray) -> float:
        """
        The objective function to be optimised. This function is called by the differential evolution. Since the DE
        tries to minimise the objective function, the fitness function is the inverse of the accuracy
        (i.e., f(x)=1 - acc) of the SSN classifier. The parameter workers of the DE is set to workers=-1
        (multiprocessing using all available CPU cores), which creates several copies of the this
        SNNClassificationTask object. Therefore, it is necessary to create a new instance of the SSN classifier and
        pass a clone (copy) of the core (attribute _core) and the number of inputs (attribute _num_inputs) as
        parameters.

        :param parameters: The agent produced by the differential evolution
        :param data: The dataset
        :param targets: The targets of the dataset
        :return: The inverse of the accuracy function (i.e., f(x)= 1 - acc)
        """
        logger.debug("Evaluating agent %s", np.array2string(parameters))
        classifier = SSNClassifier(self._num_inputs, self._core.clone())
        classifier.set_array_params(parameters)
        acc_train = self._cross_validation(classifier, data, targets, self._cv_inner)
        #print(f"Evaluating agent {np.array2string(parameters)} acc= {acc_train} \n")
        return 1 - acc_train

    def _cross_validation(self, classifier: SSNClassifier, data: list[pandas.DataFrame], targets: np.ndarray,
                          cv: StratifiedKFold = None) -> float:
        """
        Performs the stratified k-fold cross-validation over the specified dataset.

        :param classifier: A new instance of the SSNN classifier
        :param data: The dataset
        :param targets: The targets of the dataset
        :param cv: Determines the cross-validation splitting strategy. Possible inputs for cv are:
                    None, to use the default 5-fold cross validation with random data shuffle,
                    int, to specify the number of folds in a (Stratified)KFold,
        :return: The accuracy of the model
        """

        cv = check_cv(cv)
        logger.debug("Running Stratified K-fold Cross-Validation k = %s", str(cv.n_splits))
        scores = cross_val_score(classifier, data, targets, scoring='accuracy', cv=cv, n_jobs=1)
        acc = np.mean(scores)

        # predicted_values = []
        # true_values = []
        # for train_index, validation_index in skf.split(data, targets):
        #     # print("TRAIN:", train_index, "TEST:", validation_index)
        #     training_data = [data[i] for i in train_index]
        #     validation_data = [data[i] for i in validation_index]
        #     train_targets, validation_targets = targets[train_index], targets[validation_index]
        #     classifier.fit(training_data, train_targets)
        #     true_values = np.concatenate((true_values, validation_targets))
        #     predicted_values = np.concatenate((predicted_values, classifier.predict(validation_data)))
        # acc = accuracy_score(true_values, predicted_values)
        return acc

    def fit(self, data: list[pandas.DataFrame] = None,
            targets: np.ndarray = None) -> SSNClassifier:
        """
        Fits the model applying the tuned hyperparameters ('parameters'). The parameter 'parameters' must map the
        __init__ parameters (neuron_param: ndarray, weights: ndarray, firing_rates: dict) of the SSN classifier if it is
        an instance of a dict. Otherwise, if it is an instance of ndarray, it must have the values of the weights and
        the values of the neuron's parameters, and the data and targets parameters must be not None. Therefore, this
        function will fit the model with the training dataset and training targets and generate new average firing rates
        for each class.

        :param data: The dataset to use for fitting the model, usually the training set.
        :param targets: The targets of the dataset
        :return: The SSNClassifier fitted with the specified data
        """
        logger.info("Fitting the model")

        self.ssn_classifier.fit(data, targets)
        return self.ssn_classifier

    def score(self, data: list[pandas.DataFrame], targets: np.ndarray) -> float:
        """
        Evaluation of the model (SSN classifier). It predicts the labels of the data and calculates the confusion matrix
        along with the report and the accuracy score.

        :param data: The dataset to use for evaluating the model, usually the testing set.
        :param targets: Teh targets of the dataset
        :return: The accuracy fo the model
        """
        logger.info("Model evaluation")
        predicted_values = self.ssn_classifier.predict(data)
        # Printing the training accuracy
        print(metrics.confusion_matrix(targets, predicted_values))
        print(metrics.classification_report(targets, predicted_values))
        acc = accuracy_score(targets, predicted_values)
        return acc

