import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tensorflow as tf
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


def plot_confusion_matrix(cm=None, y_true=None, y_pred=None, classes=None, figsize=(10, 10), text_size=15):
    # Creates the confusion matrix
    if cm is None:
        cm = confusion_matrix(y_true, tf.round(y_pred))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:np.newaxis]
    n_classes = cm.shape[0]

    # let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    # Create classes

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # label the axes
    ax.set(title="Confusion matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)
    # set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)
    # set threshold for different colours
    threshold = (cm.max() + cm.min()) / 2

    # plot the tex on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f} %)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)

def show():
    plt.show()


def plot_iterations(file_name, num_rows: int = 0, num_cols: int = 0):
    # # Plot the Differential Evolution iterations of each fold
    # data = pd.read_csv("D:\\DataSets\\Pigeons\\Transfer learning\\Experiments\\Jason & Davison\\Cond 1\\py_results no seed\\Models\\solutions.csv")
    data = pd.read_csv(file_name)

    #    data2 = data.loc[:, data.columns != 'Acc']
    data2 = data.iloc[:, 1:-1]

    if num_rows == 0 and num_cols == 0:
        data2.T.plot(figsize=(6, 6))
    else:
        num_items = data2.shape[0]
        plt.figure(figsize=(3 * num_cols, 3 * num_rows))
        for i in range(num_items):
            plt.subplot(num_rows, num_cols, i+1)
            df = data2.iloc[i, :]
            df.plot()
            plt.xlabel("epochs={} min={:2.4f}".format(df.count(), df[df.count()-1])),
            plt.xlim([0, len(df)])
            plt.ylim([0, 0.5])
    #plt.show()


def plot_de(fitness_iteration_list: np.array = None, labels: list = None):
    plt.figure(figsize=(10,8))
    plt.plot(fitness_iteration_list)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Performance Evolution")
    if labels:
        plt.legend(labels)
    #plt.show()


def plot_raster(firing_times: np.array = None, targets: np.ndarray = None, predicted: np.ndarray = None,
                class_labels: np.array = None):

    sorted_idx = np.argsort(targets)
    unique_labels = np.unique(targets)
    palette_tab10 = sns.color_palette("tab10", len(unique_labels))
    num_samples_class = len(firing_times)/len(class_labels)

    plt.figure(figsize=(8, 6))
    counter = 0
    for row_id, index in enumerate(sorted_idx):
        y_values = np.full((len(firing_times[index]), 1), row_id)
        predicted_color_idx = palette_tab10[int(predicted[index]-1)]
        target_color_idx = palette_tab10[int(targets[index]-1)]
        marker = "o"
        if int(predicted[index]) != targets[index]:
            marker = "o"
        plt.scatter(firing_times[index], y_values, s=6.0, color=predicted_color_idx, marker=marker)
        # Draw horizontal lines to divide the classes
        if row_id % num_samples_class == 0 and row_id > 0:
            plt.axhline(y=row_id, color='black', linestyle='-', linewidth=0.25)
            counter += 1

    plt.xlabel("Time")
    plt.ylabel("Samples")
    plt.title("Raster plot")

    ax = plt.gca()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    # Reduce the box of the plot and have room for the legend box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # set the legend elements
    legend_elements = []
    for i in unique_labels:
        legend_elements.append(Line2D([0], [0], marker='o', color=palette_tab10[i - 1], label=class_labels[i - 1],
                                      markerfacecolor=palette_tab10[i - 1], markersize=4))

    ax.legend(handles=legend_elements, title="Class labels", bbox_to_anchor=(1.0, 1.0), loc='upper left', frameon=False)
    # Remove the plot borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.axis('off')    # Remove everything except the plotted line

def plot_firing_dist(data_frame: pd.DataFrame = None, x_column_name: str = "", hue_column_name: str = "",
                     class_labels: np.array = None):
    if not class_labels:
        classes = data_frame[hue_column_name].unique()
        class_labels = [("c_ " + str(i)) for i in classes]

    palette = sns.color_palette("tab10", len(class_labels))

    g = sns.displot(data=data_frame, x=x_column_name, kind="kde", hue=hue_column_name, palette=palette, fill=True)
    plt.gcf().set_size_inches(8, 6)

    # g._legend.set_title("Class labels")
    # g.fig.suptitle("Firing Density")

    # Set the class labels shown in the legend box
    for t, l in zip(g._legend.texts, class_labels):
        t.set_text(l)

    g.set(xlabel="Firing rates", ylabel="Density")
    plt.suptitle("Firing Density", y=1)
    # Move the legend
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.8, 1.0), title="Class labels")



