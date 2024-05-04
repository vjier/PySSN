# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from LIF import LIF


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def normalise_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('JIER')
    np.random.seed(42)
    # Read data
    df = pd.read_csv("D:\\DataSets\\wrist_movement_eeg\\sam2_eeg.csv", header=None)
    # Normalise data
    norm = df.sub(df.min(axis=0))
    norm = norm.div(norm.max(axis=0))

    # Create weights
    weights = np.random.uniform(-1, 1, size=(len(df.columns), ))
    current = norm.dot(weights)
    print(current)
    lif_model = LIF()
    stimulation_time = len(current)
    mp = np.zeros(stimulation_time)
    for i in range(stimulation_time):
        mp[i] = lif_model.compute_membrane_potential(i, current[i])

    print("Num firings: ", lif_model.get_num_firings())
    print("Firing rate: ", lif_model.get_firing_rate())
    print("Firing rate: ", lif_model.get_firing_rate(current))

    print(lif_model.get_firing_times())
    plt1 = plt.figure(1)
    plt.plot(mp, '-b', label="Membrane potential")
    plt.plot(current, '-r', label="Current")
    plt.eventplot(lif_model.get_firing_times(), label="Spikes")
    plt.plot([0, stimulation_time], [lif_model.get_spike_threshold(), lif_model.get_spike_threshold()], 'g--',
             label="Threshold")
    plt.legend()
    plt.show()
