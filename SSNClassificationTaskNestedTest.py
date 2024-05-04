import logging.config
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from SSNClassificationTask import SNNClassificationTask
from sklearn.model_selection import StratifiedKFold
from TemporalData import TemporalData


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


def main():
    # Data loading
    temporal_data = TemporalData("D:\\DataSets\\wrist_movement_eeg")
    dataset, targets = temporal_data.load()
    training_data, testing_data, training_targets, testing_targets = temporal_data.split()
    # Setting the parameters of the SSN Classifier task
    task = SNNClassificationTask(num_inputs=temporal_data.get_n_inputs(),
                                 max_iterations=200)

    logger.info("Running nested cross validation using the training dataset")
    outer_splits = 5
    inner_splits = 5
    random_state = 42
    cv_outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    cv_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    result = task.nested_cross_validation(training_data, training_targets, cv_outer=cv_outer, cv_inner=cv_inner)
    print(result["scores"])
    print(f"Accuracy: {result['mean_acc']:.3f}% {result['std_acc']:.3f}% ")

    df = pd.DataFrame(data=result["best_iter_fitness"])
    mean_best_iter_fitness = df.mean(axis=0)
    df['Acc'] = result["scores"]
    print(df)
    df.to_csv("D:\\DataSets\\wrist_movement_eeg\\singleNeuron\\LIF\\Nested Cross Validation\\solutions.csv")

    plt1 = plt.figure(1)
    df.T.plot(color='g')
    plt.plot(mean_best_iter_fitness, 'b')
    plt.ylabel("Iterations")
    plt.xlabel("Inverse accuracy (1-acc)")
    plt.title("Training process")
    plt.savefig("D:\\DataSets\\wrist_movement_eeg\\singleNeuron\\LIF\\Nested Cross Validation\\evolution.png",
                transparent=True)
    plt.show()


if __name__ == '__main__':
    # Logging configuration
    main()
