import numpy as np


def get_factor(target: np.ndarray, predicted: np.ndarray, delta: float = 0.002) -> float:
    """

    :param target: the target (reference) firing times
    :param predicted: the predicted (model) firing times
    :param delta: The time interval for finding a coincidence between two spike times
    :return:
    """
    if len(predicted) == 0 or len(target) == 0:
        return 0

    sampling_frequency = 1000
    n_target = len(target)
    n_predicted = len(predicted)
    n_coincidences = get_coincidences(target, predicted, delta * sampling_frequency)
    r_exp = n_predicted / sampling_frequency
    gamma = 0
    denominator = (1 - (2.0 * delta * r_exp))
    if denominator != 0:
        gamma = (2 / denominator) * ((n_coincidences - (2 * delta * n_target * r_exp)) / (n_target + n_predicted))
    return gamma


def get_coincidences(target: np.ndarray, predicted: np.ndarray, delta_bins: float) -> int:
    """
    Return the coincidences between two trains of firing times given a window time

    :param target: the target (reference) firing times
    :param predicted: the predicted (model) firing times
    :param delta_bins: the delta window (time steps)
    :return: the number of coincidences
    """
    n_coinc = 0
    j_start = 0
    for i in range(len(predicted)):
        i_coinc= 0
        f1 = predicted[i]
        for j in range(j_start, len(target)):
            f2 = target[j]
            if abs(f1-f2) <= delta_bins:
                i_coinc += 1
                j_start = j
            elif f2-f1 > delta_bins:
                break
        if i_coinc > 0:
            n_coinc += 1
    return n_coinc
