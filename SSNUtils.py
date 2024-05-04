"""
The :mod:`SSNUtils` module includes the functions that the different SSN tasks need. It also includes some functions for
plotting the results.
"""

# Author: Josafath Israel Espinosa Ramos <vjier1979@gmail.com>,
# License:

import re
import numpy as np
import plots as plots
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from SSNClassifier import SSNClassifier


def get_models(file_name: str = "", param_column: str = "", firing_column: str = "") -> list:
    """
    Return the parameters of the models saved in a comma separated CSV file. Every row in the file has the results
    of a trained SSNClassifier model: scores, mean_acc, std_acc, best_fitness, best_parameters, best_iter_fitness,
    best_firing_rates.

    :param file_name: The file that has the results of the pretrained models
    :param param_column: The column that contains the vector of weights and LIF parameters, e.g., 'best_parameters'
    :param firing_column: The column that contains the dictionary of firing rates per class, e.g.,
    'best_firing_rates'
    :return: A list of dictionaries having the main parameters of the model, eg.g., dict{'weights': array,
    'neuron_param', array, 'firing_rates_':dict}

    """

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
    return param_list


def save_confusion_matrix(cm=None, file_name: str = ""):
    """
    Save a confusion matrix into a comma separated CSV file, and into a png, pdf and eps graphics files.

    :param cm: The confusion matrix
    :param file_name: THe file name
    :return:
    """
    pd.DataFrame(cm).to_csv(file_name + ".csv")
    # Plot and save the confusion matrix
    plots.plot_confusion_matrix(cm=cm, classes=None, figsize=(8, 8), text_size=8)
    # plt.show()
    plt.savefig(file_name)
    plt.savefig(file_name + ".pdf", format="pdf")
    plt.savefig(file_name + ".eps", format="eps")


def str_array_to_num_array(data: list = None, row_list: np.ndarray = None, pattern: str = "\s+",
                           repl: str = ",") -> list:
    """
    Transform a DataFrame column of numeric arrays expressed as strings (e.g., '[0.435 0.684 ... 0.125]')
    to a list of numeric arrays (e.g., [0.435,0.684, ...,0.125]). It replaces the occurrences of
    the pattern (e.g., '\s+') in the string by the replacement repl (e.g., ','). In this example,
    any space in the string is replaced by a comma.

    :param data: The list that contains the list of arrays expressed as strings
    :param row_list: The array of selected rows or models
    :param pattern: The pattern to replace in the string
    :param repl: The string that replaces the pattern
    :return: A list of numeric arrays
    """
    fitness_iteration_list = list()
    for model_id in row_list:
        fitness_list = data[model_id]
        # fitness_array = re.sub("\s+", ",", fitness_array)
        fitness_list = re.sub(pattern, repl, fitness_list)
        fitness_list = literal_eval(fitness_list)
        fitness_iteration_list.append(np.transpose(fitness_list))
    return fitness_iteration_list


def plot_performance_evolution(model_file: str = "", column_name: str = "best_iter_fitness", model_ids: np.array = None,
                               mean: bool = False):
    """
    This function plots the best fitness value of each iteration of the Differential Evolution. It reads the  column of
    the file that contains the results of the training (DE) and select the specified models to plot.

    :param model_file: The file that has the results of the Differential Evolution
    :param column_name: The name of the column that contains the fitness of each iteration
    :param model_ids: The array of model indexes to include in the plot. If None, the figure will contain all the models
    stored in the file.
    :param mean: Set to True to show only the mean of the selected models
    :return:
    """
    models = pd.read_csv(model_file, usecols=[column_name], low_memory=True)
    model_fitness = models[column_name].to_list()
    if model_ids is None:
        model_ids = range(len(models))

    fitness_iteration_list = str_array_to_num_array(data=model_fitness, row_list=model_ids, repl="")

    if mean:
        mean_fitness = np.mean(fitness_iteration_list, axis=0)
        plots.plot_de(mean_fitness, labels=["mean"])
    else:
        labels = [("model " + str(model_id)) for model_id in model_ids]
        plots.plot_de(np.transpose(fitness_iteration_list), labels=labels)


def plot_raster(model_file: str = "", model_ids: np.array = None, param_column: str = "best_parameters",
                firings_column: str = "best_firing_rates", num_inputs: int = 0, samples: list = None,
                targets: np.ndarray = None, class_labels: np.array = None):
    """

    :param model_file:
    :param model_ids:
    :param param_column:
    :param firings_column:
    :param num_inputs:
    :param samples:
    :param targets:
    :param class_labels:
    :return:
    """
    # Step 1: Load all models contained in the results.csv file (file generated from a pretrained model)
    models = get_models(file_name=model_file, param_column=param_column, firing_column=firings_column)
    if model_ids is None:
        model_ids = range(len(models))

    firing_times_list = []
    for model_id in model_ids:
        model = SSNClassifier(num_inputs=num_inputs)
        model.set_params(**models[model_id])
        predicted_values = model.predict(samples)
        for sample in samples:
            firing_times = model.get_firing_times(sample=sample)
            firing_times_list.append(np.array(firing_times))
        plots.plot_raster(firing_times=firing_times_list, targets=targets, predicted=predicted_values,
                          class_labels=class_labels)


def plot_confusion_matrix(model_file: str = "", model_ids: np.array = None, param_column: str = "best_parameters",
                          firings_column: str = "best_firing_rates", num_inputs: int = 0, samples: list = None,
                          targets: np.ndarray = None, class_labels: np.array = None, figsize=(10, 10), text_size=15):
    # Step 1: Load all models contained in the results.csv file (file generated from a pretrained model)
    models = get_models(file_name=model_file, param_column=param_column, firing_column=firings_column)
    if model_ids is None:
        model_ids = range(len(models))

    firing_times_list = []
    for model_id in model_ids:
        model = SSNClassifier(num_inputs=num_inputs)
        model.set_params(**models[model_id])
        predicted_values = model.predict(samples)
        plots.plot_confusion_matrix(y_true=targets, y_pred=predicted_values, classes=class_labels, figsize=figsize,
                                    text_size=text_size)


def plots_show():
    plots.show()


def save_num_firings(model_file: str = "", model_ids: np.array = None, param_column: str = "best_parameters",
                     firings_column: str = "best_firing_rates", num_inputs: int = 0, samples: list = None,
                     targets: np.ndarray = None, class_labels: np.array = None):
    models = get_models(file_name=model_file, param_column=param_column, firing_column=firings_column)
    if model_ids is None:
        model_ids = range(len(models))

    firing_times_list = []
    for model_id in model_ids:
        model = SSNClassifier(num_inputs=num_inputs)
        model.set_params(**models[model_id])
        predicted_values = model.predict(samples)
        for i, sample in enumerate(samples):
            if i == 133:
                print("here")
            firing_times = model.get_firing_times(sample=sample)
            firing_times_list.append(len(firing_times))

        int_pred_values = [int(i) for i in predicted_values]

        results = {"firing_times": firing_times_list,
                   "targets": targets,
                   "predicted": int_pred_values,
                   }
        pd.DataFrame(results).to_csv("D:/num_firing_times_model_" + str(model_id) + ".csv", index=False)


def save_firing_rates(model_file: str = "", model_ids: np.array = None, param_column: str = "best_parameters",
                      firings_column: str = "best_firing_rates", num_inputs: int = 0, samples: list = None,
                      targets: np.ndarray = None, class_labels: np.array = None):
    models = get_models(file_name=model_file, param_column=param_column, firing_column=firings_column)
    if model_ids is None:
        model_ids = range(len(models))

    firing_rates_list = []
    for model_id in model_ids:
        model = SSNClassifier(num_inputs=num_inputs)
        model.set_params(**models[model_id])
        predicted_values = model.predict(samples)
        for i, sample in enumerate(samples):
            if i == 133:
                print("here")
            firing_rate = model._get_firing_rate(sample)
            firing_rates_list.append(firing_rate)

        int_pred_values = [int(i) for i in predicted_values]

        results = {"firing_rates": firing_rates_list,
                   "targets": targets,
                   "predicted": int_pred_values,
                   }
        pd.DataFrame(results).to_csv("D:/num_firing_rates_model_" + str(model_id) + ".csv", index=False)


def plot_firing_distribution(model_file: str = "", model_ids: np.array = None, param_column: str = "best_parameters",
                             firings_column: str = "best_firing_rates", num_inputs: int = 0, samples: list = None,
                             class_labels: np.array = None):
    models = get_models(file_name=model_file, param_column=param_column, firing_column=firings_column)
    if model_ids is None:
        model_ids = range(len(models))

    firing_rates_list = []
    for model_id in model_ids:
        model = SSNClassifier(num_inputs=num_inputs)
        model.set_params(**models[model_id])
        predicted_values = model.predict(samples)
        for sample in samples:
            #firing_times = model.get_firing_times(sample=sample)
            #firing_times_list.append(len(firing_times))
            firing_rate = model._get_firing_rate(sample)
            firing_rates_list.append(firing_rate)

        int_pred_values = [int(i) for i in predicted_values]

        results = {"firing_rates": firing_rates_list,
                   "class_labels": int_pred_values,
                   }
        df = pd.DataFrame(results)

        if not class_labels:
            classes = df["class_labels"].unique()
            class_labels = [("c_ " + str(i)) for i in classes]

        plots.plot_firing_dist(data_frame=df, x_column_name="firing_rates", hue_column_name="class_labels",
                               class_labels=class_labels)
