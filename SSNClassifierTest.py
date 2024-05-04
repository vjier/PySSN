import logging.config
import os
import fnmatch
import natsort
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from SSNClassifier import SSNClassifier
from TemporalData import TemporalData


from sklearn.metrics import precision_score


def main():
    temporal_data = TemporalData("D:\\DataSets\\wrist_movement_eeg")
    samples, targets = temporal_data.load()
    # SSN Classifier configuration
    # The following parameters achieved an accuracy acc = 93% during training and acc = 100 % during the testing stages,
    # resulting in an overall acc = 95%. However, these values may vary depending on how the data for training and
    # testing are split.
    model = SSNClassifier(temporal_data.get_n_inputs())
    # parameters = {"neuron_param": np.array([2.39084394e+00,  1.91238713e+00]),
    #               "weights": np.array([2.09772479e-03, -2.71616645e-03, 5.07892017e-03, -3.59378264e-04,
    #                                    2.68028572e-03, -4.33388668e-03, -8.19900908e-03, -7.66850340e-03,
    #                                    -1.70152903e-03, 8.18406986e-03, 2.01867698e-03, -4.44234413e-06,
    #                                    5.80681358e-03, 1.15362468e-03]),
    #               "firing_rates_": {1: 0.2651285320517731, 2: 0.3317432987543224, 3: 0.31732810065408484}
    #               }

    parameters = {"neuron_param": np.array([0.40126242298205483,2.325981595685085]),
                  "weights": np.array([0.006470007071424052,-0.005096412979488722,0.0025566008715812204,-9.766390137978347E-4,0.007132691561097668,-0.0017022162620066166,-0.007113840140167936,-0.009120016184370899,-0.002182167380514029,0.00730643792703817,-0.004886484118318466,0.006850923999251521,0.007855736901826608,-0.0077742993693509874]),
                  "firing_rates_": {1:0.2080078125, 2:0.2490234375, 3:0.23876953125}
                  }

    # 0.009095424024688659,-0.002900899506259171,0.007864814137408771,-0.009081002557789318,0.0056379920690531,-0.009415047214156995,-0.009054146172294867,-0.0070794129556275565,-6.818661217588168E-5,0.0016059503412939732,0.006332595542752785,-0.0012527335414026818,0.009453366780583842,-0.003983533869277529,0.37204639698466857,1.3592309015775739
    model.set_params(**parameters)
    model.fit(samples, targets)

    acc = model.score(samples, targets)
    print("Acc: ", acc)

    filename = "D:\\SSNClassifier.ssn"
    pickle.dump(model, open(filename, 'wb'))

    # Prediction
    predicted_values = model.predict(samples)
    # Model evaluation
    logger.info("Model attributes: %s", str(model.get_attributes()))
    print(metrics.confusion_matrix(targets, predicted_values))
    print(metrics.classification_report(targets, predicted_values))
    acc = accuracy_score(targets, predicted_values)
    print("Acc: ", acc)

    # Prediction loaded model
    loaded_model = pickle.load(open(filename, 'rb'))
    logger.info("Loaded model attributes: %s", str(loaded_model.get_attributes()))
    predicted_values = loaded_model.predict(samples)
    print(metrics.confusion_matrix(targets, predicted_values))
    print(metrics.classification_report(targets, predicted_values))
    acc = accuracy_score(targets, predicted_values)
    print("Acc: ", acc)


    # Raster plot
    plt1 = plt.figure(1)
    for i in range(len(samples)):
        vec = model.get_firing_times(samples[i])
        if targets[i] != predicted_values[i]:
            color = [1, 0, 0]
        else:
            color = [0, 0, 1]
        a = np.ones(len(vec))*i
        plt.scatter(vec, a, s=1.5, color=color)
        plt.ylabel("Samples")
        plt.xlabel("Firing times")
        plt.title("Classification raster plot")
    plt.show()


if __name__ == '__main__':
    # Logging configuration
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('default')
    main()

