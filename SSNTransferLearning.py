import re
import os
import logging.config
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from ast import literal_eval
from parameters import parameter_parser
from TemporalData import TemporalData
from SSNClassifier import SSNClassifier
from SSNClassificationTask import SNNClassificationTask

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


def get_models(file_name: str = "", param_column: str = "", firing_column: str = "") -> list:
    """
    Return the parameters of the models saved in a  comma separated CSV file. Every row in the file has the results
    of a trained SSNClassifier model: scores, mean_acc, std_acc, best_fitness, best_parameters, best_iter_fitness,
    best_firing_rates.

    :param file_name: The file that has the results of the pretrained models
    :param param_column: The column that contains the vector of weights and LIF parameters, e.g., 'best_parameters'
    :param firing_column: The column that contains the dictionary of firing rates per class, e.g., 'best_firing_rates'
    :return: A list of dictionaries having the main parameters of the model, eg.g., dict{'weights': array,
    'neuron_param', array, 'firing_rates_':dict}

    """
    logger.info("Loading models from " + file_name)
    data = pd.read_csv(file_name)
    num_rows = data.shape[0]
    param_list = list()
    for model_id in range(num_rows):
        snn_parameters = data.iloc[model_id, data.columns.get_loc(param_column)]
        firing_rates = data.iloc[model_id, data.columns.get_loc(firing_column)]
        firing_rates = literal_eval(firing_rates)
        snn_parameters = re.sub("\s+", ",", snn_parameters)
        snn_parameters = literal_eval(snn_parameters)
        parameters = {"weights": np.array(snn_parameters[:2]),
                      "neuron_param": np.array(snn_parameters[2:]),
                      "firing_rates_": firing_rates
                      }
        param_list.append(parameters)
    logger.info("Number of models loaded: " + str(num_rows))
    return param_list


def create_initial_population(model: dict = None, pop_size: int = 10, pop_dist: str = "", min: float = 0.0,
                              max: float = 1.0, std_dev: float = 0.5, seed: int = None):
    """
    Create a random population for the Differential Evolution based on the parameters of the selected model.

    :param model: A dictionary with at least the weights and neuron parameters of the model, e.g., dic{'weights':array,
    'neuron_param':array}
    :param pop_size: A multiplier for setting the total population size. The population has pop_size * d individuals,
    where d is the dimensionality of an individual (agent or candidate solution). Pop_size must be less than the number
    of rows in the specified column".
    :param pop_dist: The probability distribution strategy to create the initial population of the Differential
    Evolution for retraining the model. This strategy should be one of: 'uniform', 'bounded, 'normal'.
    :param min: A scalar or array indicating the minimum value of each dimension of a candidate solution of the
    Differential Evolution.
    :param max: A scalar or array indicating the maximum value of each dimension of a candidate solution of the
    Differential Evolution.
    :param std_dev: A scalar or array indicating the standard deviation of each dimension of an agent of the
    Differential Evolution.
    :param seed: The seed generator for creating a random population.
    :return:
    """
    logger.info("Creating initial population from model")
    # An agent of the DE is composed by the weights and neuron parameters
    patterns = np.append(model["weights"], model["neuron_param"])
    # Get the dimensionality of the problem (agents' size)
    dim = len(patterns)

    # Creates the new random population of size (popsize * dimensionality) - number of elements in the specified column
    np.random.seed(seed=seed)
    if pop_dist == 'uniform':
        logger.info("Creating initial population with uniform distribution")
        population = np.random.uniform(size=((pop_size * dim) - 1, dim))
    elif pop_dist == "bounded":
        logger.info("Creating initial population within boundaries")
        population = np.random.uniform(low=min, high=max, size=((pop_size * dim) - 1, dim))
    elif pop_dist == "normal":
        logger.info("Creating initial population with normal distribution")
        population = np.random.normal(loc=patterns, scale=std_dev, size=((pop_size * dim) - 1, dim))

    # Add the model to the population
    population = np.append([patterns],population , axis=0)
    return population


def feature_extraction(model: SSNClassifier, samples, targets, args):
    """
    Feature extraction transfer learning takes the underlying patterns (weights and neuron's parameters) a pretrained
    model has learned and adjust its outputs (firing rates) to be more suited to the data. It always creates new firing
    rates which which can be calculated using the whole dataset. However, the best practice is to perform the k-fold
    cross-validation option where the firing rates are calculated in every fold.

    :param model: A new SSNClassifier model to be trained.
    :param samples: The dataset to be trained.
    :param targets: The labels of the dataset.
    :param args: A set of parameters necessary for performing the transfer learning experiments.
    :return: A dictionary dict{'scores': list, 'mean_acc': float, 'std_acc': float, 'weights': list(array),
    'neuron_param': list(array), 'firing_rates_': dict{int:float}}.
    """
    logger.info("Performing " + args.transfer_type)

    # Create a new instance of the SSN classifier

    if args.cv:
        # Run a stratified k-fold cross validation to evaluate the performance of the model on the new dataset.
        cv = StratifiedKFold(n_splits=args.cv_num_splits, shuffle=True, random_state=args.cv_seed)
        # Executes the cross-validation method
        scores = cross_val_score(model, samples, targets, scoring=args.cv_scoring, cv=cv, n_jobs=1)
    else:
        # Create new firing rates from the whole dataset and make predictions using the whole dataset too.
        scores = [model.evaluate(samples=samples, targets=targets, scoring=accuracy_score)]

    acc = np.mean(scores)
    std = np.std(scores)

    results = {"scores": list(scores),
               "mean_acc": acc,
               "std_acc": std,
               "weights": list(model.weights),
               "neuron_param": list(model.neuron_param),
               "firing_rates_": sorted(model.firing_rates_.items())
               }
    return results


def fine_tuning(model: SSNClassifier, num_inputs, base_model, samples, targets, args):
    """
    Fine tuning transfer learning takes the underlying patterns (weights and neuron's parameters) of a pretrained model
    and adjust (fine-tune, train) them to the problem. New firing rates are generated.

    :param model: A new SSNClassifier model to be trained.
    :param num_inputs: Number of features of the temporal data.
    :param base_model: A dictionary with at least the weights and neuron parameters of the model, e.g.,
    dic{'weights':array, 'neuron_param':array}.
    :param samples: The dataset to be trained.
    :param targets: The labels of the dataset.
    :param args: A set of parameters necessary for performing the transfer learning experiments.
    :return: A dictionary dict{'scores': list, 'mean_acc': float, 'std_acc': float, 'weights': list(array),
    'neuron_param': list(array), 'firing_rates_': dict{int:float}}.

    """
    #
    logger.info("Performing " + args.transfer_type)
    # Create a population from a model
    initial_population = create_initial_population(model=base_model, pop_size=args.pop_size, pop_dist=args.pop_dist,
                                                   min=args.min_val, max=args.max_val, std_dev=args.std_dev,
                                                    seed=args.pop_seed)
    print(initial_population)
    # Create a new classification task for retraining the model
    task = SNNClassificationTask(snn_classifier=model, num_inputs=num_inputs,
                                 max_iterations=args.max_iterations, initial_population=initial_population)
    # Define the outer cross-validation method
    cv_outer = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.nested_seed)
    # Define the inner cross-validation method
    cv_inner = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=args.nested_seed)
    results = task.nested_cross_validation(samples, targets, cv_outer=cv_outer, cv_inner=cv_inner)
    return results


if __name__ == "__main__":
    # Logs configuration
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('default')

    # Read parameters fro experiments
    args = parameter_parser()

    # Load the temporal data samples
    temporal_data = TemporalData(args.input_path)
    samples, targets = temporal_data.load(display=True)
    num_inputs = temporal_data.get_n_inputs()

    # Load models (SNN parameters and firing rates) from a file. The file must be generated from the classification
    # task using nested cross validation.
    models = get_models(file_name=args.model_file, param_column=args.param_column, firing_column=args.firings_column)
    model_ids = args.model_ids
    if model_ids is None:
        model_ids = range(len(models))

    # Start transfer learning
    logger.info("Transfer learning modality: " + args.transfer_type)
    result_list = list()

    # Run the transfer learning from each pretrained model
    for id in model_ids:
        results_dir = args.output_path + os.path.sep + args.transfer_type
        results_file_name = "results_transfer_" + args.transfer_type
        # Select a model
        base_model = models[id]
        logger.info("Transferring model " + str(id) + " with parameters: " + str(base_model))

        # Create new model
        new_model = SSNClassifier(num_inputs)
        # Transfer learning
        new_model.set_params(**base_model)

        if args.transfer_type == "feature_extraction":
            # feature_extraction: take the underlying patterns (also called weights) a pretrained model has learned and
            # adjust its outputs (firing rates) to be more suited to the problem.
            results = feature_extraction(model=new_model, samples=samples, targets=targets, args=args)
            if args.cv:
                results_file_name = results_file_name+"_cv.csv"
            else:
                results_file_name = results_file_name + "_all.csv"
        else:
            # fine_tuning: This type of transfer learning takes the underlying patterns (weights and neuron parameters)
            # of the pretrained model and adjust (fine-tune, train) them to the problem.
            results = fine_tuning(model=new_model, num_inputs=num_inputs, base_model=base_model, samples=samples, targets=targets, args=args)
            results_dir = results_dir + os.path.sep + args.pop_dist
            results_file_name = results_file_name + "_" + args.pop_dist+".csv"

        result_list.append(results)
    # Save the results to a csv file through pandas DataFrame
    df = pd.DataFrame(result_list)
    print(df)
    df.to_csv(results_dir + os.path.sep + results_file_name, index=False)

