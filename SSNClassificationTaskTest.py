import logging.config
import numpy as np
import matplotlib.pyplot as plt
from SSNClassificationTask import SNNClassificationTask
from TemporalData import TemporalData


def main():
    # Data loading
    temporal_data = TemporalData("D:\\DataSets\\wrist_movement_eeg")
    dataset, targets = temporal_data.load()
    training_data, testing_data, training_targets, testing_targets = temporal_data.split()

    # Setting the parameters of the SSN Classifier task
    task = SNNClassificationTask(temporal_data.get_n_inputs(), max_iterations=200, n_outer_splits=10, n_inner_splits=5)

    # logger.info("Running nested cross validation using the training dataset")
    # nested_results = task.nested_cross_validation(training_data, training_targets)
    # print(nested_results["acc"])

    logger.info("Tuning the model's hyperparameters using the training dataset")
    # Fitting the model with the training dataset applying the tuned hyperparameters
    parameters = task.tune(training_data, training_targets)
    logger.info("Fitting the model")
    model = task.fit(parameters.x, training_data, training_targets)

    logger.info("Model attributes: %s", str(model.get_attributes()))
    # Evaluate the model using the training dataset. Just to know the accuracy during the training stage.
    logger.info("Evaluating the model using the training dataset (training accuracy)")
    skill_train = task.score(training_data, training_targets)
    logger.info("Acc train: %s", str(skill_train))
    # Evaluate the model using the testing dataset.
    logger.info("Evaluating the model using the testing dataset (generalisation)")
    skill_test = task.score(testing_data, testing_targets)
    logger.info("Acc test: %s", str(skill_test))

    # Raster plot
    firing_times = []
    samples = temporal_data.get_samples()
    for i in range(len(samples)):
        vec = model.get_firing_times(samples[i])
        firing_times.append(vec.copy())

    plt1 = plt.figure(1)
    plt.eventplot(firing_times, linelengths=0.3)
    plt.ylabel("Samples")
    plt.xlabel("Firing times")
    plt.title("Classification raster plot")

    # Raster plot showing misclassified
    predicted_values = model.predict(samples)
    plt2 = plt.figure(2)
    for i in range(len(samples)):
        vec = model.get_firing_times(samples[i])
        if targets[i] != predicted_values[i]:
            color = [1, 0, 0]
        else:
            color = [0, 0, 1]
        a = np.ones(len(vec)) * i
        plt.scatter(vec, a, s=1.5, color=color)
        plt.ylabel("Samples")
        plt.xlabel("Firing times")
        plt.title("Classification raster plot")

    # Plotting the training (evolution) process of the classifier
    plt3 = plt.figure(3)
    plt.plot(parameters.best_agent_values)
    plt.ylabel("Iterations")
    plt.xlabel("Inverse accuracy (1-acc)")
    plt.title("Training process")
    plt.show()


if __name__ == '__main__':
    # Logging configuration
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('default')
    main()
