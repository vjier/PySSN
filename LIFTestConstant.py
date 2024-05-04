import math
import numpy as np
import matplotlib.pyplot as plt
from LIF import LIF

if __name__ == '__main__':
    """
        Simulation of a Leaky Integrate-and-fire model wit a constant current
    """
    lif_model = LIF(spike_threshold=15, absolute_refractory_period=0)
    current = 20
    stimulation_time = 100
    mp = np.zeros(stimulation_time)
    for i in range(stimulation_time):
        mp[i] = lif_model.compute_membrane_potential(i, current)

    print("Num firings: ", lif_model.get_num_firings())
    print("Firing rate with time: ", lif_model.get_firing_rate())
    print("Firing rate formula: ", lif_model.get_firing_rate(current))

    plt1 = plt.figure(1)
    plt.plot(mp, '-b')

    delta_abs = 5
    tau_m = 10
    v_th = 15
    Is = np.linspace(1, 50, 51)
    x = Is / (Is - v_th)
    y = np.log(x, where=0 < x, out=np.nan * x)
    f1 = 1.0 / (0 + tau_m * y)
    f1[np.isnan(f1)] = 0
    f2 = 1.0 / (delta_abs + tau_m * y)
    f2[np.isnan(f2)] = 0
    print(len(Is))
    print(len(f1))
    plt2 = plt.figure(2)
    plt.rcParams['lines.linewidth'] =1.0
    plt.plot(Is, f1, 'k-')
    plt.plot(Is, f2, 'k--')
    plt.plot([current, current], [0, 0.3], 'r-')
    plt.xlim([0, 50])
    plt.xlabel('I', fontsize=24)
    plt.ylabel('f(kHz)', fontsize=24)
    plt.yticks([0, .1, .2, .3])
    plt.show()


