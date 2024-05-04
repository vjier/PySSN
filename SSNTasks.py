"""
The :mod:`SSNTask` module includes a class and functions to run different task on a single spiking neural network
architecture.
"""

# Author: Josafath Israel Espinosa Ramos <vjier1979@gmail.com>,
# License:

import os
import logging.config

from parameters import parameter_parser

from SSNClassifier import SSNClassifier
from SSNClassificationTask import SNNClassificationTask
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import metrics

import SSNUtils as utils
import numpy as np
import pandas as pd

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


class SNNTasks:
    """
    The SNNTask class has the functions to execute the train, recall and the feature extraction and fine
    tuning transfer learning of a single spiking neural network (SNN) architecture. All the functions receive a
    parameter parser which contains the arguments and values to perform a specific task.
    """

    def train(self, num_inputs: int = 0, samples: list = None, targets: np.ndarray = None,
              args: parameter_parser = None):
        """
        The train function creates a new SNNClassifier model using a nested cross-validation method. The nested cross
        validation split the samples into training and test datasets.

        :param num_inputs: The number of inputs of the SNN, i.e., the number of temporal features (columns of a sample)
        :param samples: The temporal dataset used to create the model. The list of matrices where the rows and columns of
        each matrix represent the temporal points and features respectively.
        :param targets: The list of sample labels
        :param args: The arguments or parameter used to train a model (max_iterations, outer_splits, inner_splits,
        nested_seed, output_path)
        :return: Saves the result in a solutions.csv (the best fitness value of each iteration of the Differential
        Evolution), mean_best_iter_fitness.csv (the mean fitness value of all outer splits of the cross-validation at
        each iteration of the Differential Evolution), best_firing_rates.csv (the firing rates of the best solution of
        each outer split of  the cross-validation) and results.csv (the summary of each outer split of the nested
        cross-validation that includes scores, mean_acc, std_acc, best_fitness, best_parameters, best_iter_fitness,
        best_firing_rates) files. The results.csv is used in the recall (classification of new dataset) and transfer
        learning.
        """
        logger.info("Training the SSN in a nested cross-validation fashion.")
        # Step 1: Create the model
        model = SSNClassifier(num_inputs, weight_bounds=[-10.0, 10.0])

        # Step 2: Set the parameters of the SSN Classifier task
        # Set the boundaries of the initial weight values. This parameter is crucial for achieving good performance.
        task = SNNClassificationTask(snn_classifier=model, num_inputs=num_inputs, max_iterations=args.max_iterations)

        # Step 3: Set the parameters of the validation model
        # Defines the outer cross-validation method
        cv_outer = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.nested_seed)
        # Defines the outer cross-validation method
        cv_inner = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=args.nested_seed)

        # Step 4:  Executes the nested cross-validation method
        logger.info("Running nested cross validation using the training dataset")
        results = task.nested_cross_validation(samples, targets, cv_outer=cv_outer, cv_inner=cv_inner)

        # Step 5: Show and save the results in a comma separated CSV file
        results_dir = args.output_path
        pd.DataFrame(results).to_csv(results_dir + os.path.sep + "results.csv")
        print(results["scores"])
        print(f"Accuracy: {results['mean_acc']:.3f}% {results['std_acc']:.3f}% ")
        df = pd.DataFrame(data=results["best_iter_fitness"])
        mean_best_iter_fitness = df.mean(axis=0)
        df['Acc'] = results["scores"]
        print(df)

        # Save the best solution fitness of each differential evolution iteration for each outer split
        df.to_csv(results_dir + os.path.sep + "solutions.csv")
        # Save the best solution fitness average of each differential evolution iteration over all outer splits
        pd.DataFrame(mean_best_iter_fitness).to_csv(results_dir + os.path.sep + "mean_best_iter_fitness.csv")
        # Save the firing rates of the best solution
        pd.DataFrame(results["best_firing_rates"]).to_csv(results_dir + os.path.sep + "best_firing_rates.csv")
        # Save the result object (see the return parameter of nested cross validation) to a comma separated CSV file.
        # This file is read in the recall and transfer learning
        pd.DataFrame(results).to_csv(results_dir + os.path.sep + "results.csv")

    def recall(self, num_inputs: int = 0, samples: list = None, targets: np.ndarray = None,
               args: parameter_parser = None):
        """
        The recall function classifies the dataset. It loads the parameters of pretrained models from a comma separated
        CSV file (see the results.csv file created at the end of the train() function).

        :param num_inputs:  The number of inputs of the SNN, i.e., the number of temporal features (columns of a sample)
        :param samples: The temporal dataset to classify. The list of matrices where the rows and columns of each matrix
        represent the temporal points and features respectively.
        :param targets: The list of sample labels
        :param args: The arguments or parameter used to train a model (model_file, param_column, firings_column,
        model_ids, output_path)
        :return: Save the confusion matrix in png, pdf, eps and csv files, the classification report in the
        classification_report.csv file and the accuracy results in the recall_results.csv file of each model specified
        in the model_ids argument.
        """
        logger.info("Data classification")
        # Step 1: Load all models contained in the results.csv file (file generated from a pretrained model)
        logger.info("Loading models from " + args.model_file)
        models = utils.get_models(file_name=args.model_file, param_column=args.param_column,
                                  firing_column=args.firings_column)
        logger.info("Number of models loaded: " + str(len(models)))
        model_ids = args.model_ids
        if model_ids is None:
            model_ids = range(len(models))

        result_list = list()
        # Classify data using each pretrained model
        results_dir = args.output_path
        for model_id in model_ids:
            model = SSNClassifier(num_inputs=num_inputs)
            model.set_params(**models[model_id])
            predicted_values = model.predict(samples)
            logger.info("Calculating confusion matrix")
            file_name = results_dir + os.path.sep + "confusion_matrix_" + str(model_id)
            confusion_matrix = metrics.confusion_matrix(targets, predicted_values)
            utils.save_confusion_matrix(cm=confusion_matrix, file_name=file_name)

            # print(metrics.confusion_matrix(targets, predicted_values))
            logger.info("Creating classification report")
            report = metrics.classification_report(targets, predicted_values, output_dict=True)
            file_name = results_dir + os.path.sep + "classification_report_" + str(model_id) + ".csv"
            pd.DataFrame(report).transpose().to_csv(file_name, index=False)
            # print(metrics.classification_report(targets, predicted_values))
            acc = metrics.accuracy_score(targets, predicted_values)
            results = {"scores": acc,
                       "mean_acc": acc,
                       "std_acc": 0,
                       "weights": list(model.weights),
                       "neuron_param": list(model.neuron_param),
                       "firing_rates_": sorted(model.firing_rates_.items())
                       }
            result_list.append(results)

        df = pd.DataFrame(result_list)
        df.to_csv(results_dir + os.path.sep + "recall_results.csv", index=False)

    def transfer(self, num_inputs: int = 0, samples: list = None, targets: np.ndarray = None,
                 args: parameter_parser = None):
        """
        The transfer function can execute two approaches of transfer learning: 1) feature selection and 2) fine tuning.

        :param num_inputs:  The number of inputs of the SNN, i.e., the number of temporal features (columns of a sample)
        :param samples: The temporal dataset to classify. The list of matrices where the rows and columns of each matrix
        represent the temporal points and features respectively.
        :param targets: The list of sample labels
        :param args: The arguments or parameter used to train a model (model_file, param_column, firings_column,
        model_ids, transfer_type, cv, pop_dist, output_path)
        :return: Saves the result in the results.csv file which has the summary of each outer split of the nested
        cross-validation including scores (same as mean acc), mean_acc, std_acc, best_fitness, best_parameters,
        best_iter_fitness, best_firing_rates.
        """
        logger.info("Executing transfer learning modality: " + args.transfer_type)
        # Load models (SNN parameters and firing rates) from a file. The file must be generated from the classification
        # task using nested cross validation.
        logger.info("Loading models from " + args.model_file)
        models = utils.get_models(file_name=args.model_file, param_column=args.param_column,
                                  firing_column=args.firings_column)
        logger.info("Number of models loaded: " + str(len(models)))
        model_ids = args.model_ids

        if model_ids is None:
            model_ids = range(len(models))

        results_dir = args.output_path + os.path.sep + args.transfer_type
        results_file_name = "results_transfer_" + args.transfer_type
        result_list = list()
        if args.transfer_type == "feature_extraction":
            # feature_extraction: take the underlying patterns (also called weights) a pretrained model has learned
            # and adjust its outputs (firing rates) to be more suited to the problem.
            for model_id in model_ids:
                # Select a model
                base_model = models[model_id]
                logger.info("Transferring model " + str(model_id) + " with parameters: " + str(base_model))
                # Validate the number of inputs are the same as the weights
                if num_inputs != len(base_model["weights"]):
                    logger.error(f"The number of data inputs ({num_inputs}) is different from the number of weights "
                                 f"({len(base_model.weights)}).")
                    raise Exception(f"The number of data inputs ({num_inputs}) is different from the number of weights "
                                    f"({len(base_model.weights)}). ")

                # Create new model
                new_model = SSNClassifier(num_inputs)
                # Transfer learning
                new_model.set_params(**base_model)
                results = self.feature_extraction(model=new_model, samples=samples, targets=targets, args=args)

                result_list.append(results)
            if args.cv:
                results_file_name = results_file_name + "_cv.csv"
            else:
                results_file_name = results_file_name + "_all.csv"
        elif args.transfer_type == "fine_tuning":
            # fine_tuning: This type of transfer learning takes the underlying patterns (weights and neuron
            # parameters) of the pretrained model and adjust (fine-tune, train) them to the problem.

            base_model_list = list()
            # select the models which are the base for creating the initial population
            for model_id in model_ids:
                base_model = models[model_id]
                if num_inputs != len(base_model["weights"]):
                    logger.error(f"The number of data inputs ({num_inputs}) is different from the number of weights "
                                 f"({len(base_model.weights)}).")
                    raise Exception(f"The number of data inputs ({num_inputs}) is different from the number of weights "
                                    f"({len(base_model.weights)}). ")
                base_model_list.append(base_model)
            # Create new model
            new_model = SSNClassifier(num_inputs)
            results = self.fine_tuning(model=new_model, num_inputs=num_inputs, base_models=base_model_list,
                                       samples=samples, targets=targets, args=args)

            results_file_name = results_file_name + "_" + args.pop_dist + ".csv"
            result_list.append(results)

        # Save the results to a csv file through pandas DataFrame
        df = pd.DataFrame(result_list)
        print(df)
        df.to_csv(results_dir + os.path.sep + results_file_name, index=False)

    def feature_extraction(self, model: SSNClassifier = None, samples: list = None, targets: np.ndarray = None,
                           args: parameter_parser = None):
        """
        Feature extraction transfer learning takes the underlying patterns (weights and neuron's parameters) a
        pretrained model has learned and adjust its outputs (firing rates) to be more suited to the data. It always
        creates new firing rates and a cross-validation scheme must be implemented.

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
            # Fit the new model using the
            scores = [model.evaluate(samples=samples, targets=targets, scoring=metrics.accuracy_score)]

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

    def fine_tuning(self, model: SSNClassifier = None, num_inputs: int = 0, base_models: list = None,
                    samples: list = None, targets: np.ndarray = None, args: parameter_parser = None):
        """
        Fine tuning transfer learning takes the underlying patterns (weights and neuron's parameters) of a pretrained
        model and adjust (fine-tune, train) them to the problem. New firing rates are generated.

        :param model: A new SSNClassifier model to be trained.
        :param num_inputs: Number of features of the temporal data.
        :param base_models: A list of dictionaries with at least the weights and neuron parameters of the model, e.g.,
        dic{'weights':array, 'neuron_param':array}. THe list is used for creating an initial population of the DE.
        :param samples: The dataset to be trained.
        :param targets: The labels of the dataset.
        :param args: A set of parameters necessary for performing the transfer learning experiments.
        :return: A dictionary dict{'scores': list, 'mean_acc': float, 'std_acc': float, 'weights': list(array),
        'neuron_param': list(array), 'firing_rates_': dict{int:float}}.

        """
        #
        logger.info("Performing " + args.transfer_type)
        # Create a population from a model
        initial_population = self.create_initial_population(models=base_models, pop_size=args.pop_size,
                                                            pop_dist=args.pop_dist, min_val=args.min_val,
                                                            max_val=args.max_val, seed=args.pop_seed)
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

    def create_initial_population(self, models: list = None, pop_size: int = 10, pop_dist: str = "",
                                  min_val: float = 0.0, max_val: float = 1.0, seed: int = None):
        """
        Create a random population for the Differential Evolution based on the parameters of the selected model.

        :param models: A list of dictionaries with at least the weights and neuron parameters of the model, e.g.,
        dic{'weights':array, 'neuron_param':array}. The list is used to create the initial population
        :param pop_size: A multiplier for setting the total population size. The population has pop_size * d individuals
        where d is the dimensionality of an individual (agent or candidate solution). Pop_size must be less than the
        number of rows in the specified column".
        :param pop_dist: The probability distribution strategy to create the initial population of the Differential
        Evolution for retraining the model. This strategy should be one of: 'uniform', 'bounded, 'normal'.
        :param min_val: A scalar or array indicating the minimum value of each dimension of a candidate solution of the
        Differential Evolution.
        :param max_val: A scalar or array indicating the maximum value of each dimension of a candidate solution of the
        Differential Evolution.
        :param seed: The seed generator for creating a random population.
        :return:
        """
        logger.info("Creating initial population from model")
        # An agent of the DE is composed by the weights and neuron parameters
        pattern_list = []
        for model in models:
            pattern_list.append(np.append(model["weights"], model["neuron_param"]))
        #        patterns = np.append(model["weights"], model["neuron_param"])
        # Get the dimensionality of the problem (agents' size)
        dim = len(pattern_list[0])

        # Creates the new random population of size (pop_size * dimensionality) - number of elements in the specified
        # column
        np.random.seed(seed)
        population = []
        if pop_dist == 'uniform':
            logger.info("Creating initial population with uniform distribution")
            population = np.random.uniform(size=((pop_size * dim) - len(pattern_list), dim))
        elif pop_dist == "bounded":
            logger.info("Creating initial population within boundaries")
            population = np.random.uniform(low=min_val, high=max_val, size=((pop_size * dim) - len(pattern_list), dim))
        elif pop_dist == "normal":
            logger.info("Creating initial population with normal distribution")
            mu = np.mean(pattern_list, axis=0)
            std = np.std(pattern_list, axis=0)
            population = np.random.normal(loc=mu, scale=std, size=((pop_size * dim) - len(pattern_list), dim))

        # Add the model to the population
        population = np.append(pattern_list, population, axis=0)
        return population
