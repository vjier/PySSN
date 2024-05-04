import logging.config
import os
import fnmatch
import natsort
import pandas
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Logging configuration
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('default')


class TemporalData:

    _samples = list()
    _targets = []
    _n_inputs = 0
    _class_labels = {}

    def __init__(self, data_dir: str, file_prefix: str = "sam*", file_extension: str = "csv",
                 targets_file_name: str = "tar_class.csv"):
        self._data_dir = data_dir
        self._file_prefix = file_prefix
        self._file_extension = file_extension
        self._targets_file_name = targets_file_name

    def load(self, scale: bool = False, display: bool = False) -> tuple[list[pandas.DataFrame], np.ndarray]:
        """
        loads the dataset and the file containing the class labels.

        :param scale: Set to true if the elements of the samples must be scaled in the range 0 to 1
        :param display: Display the names of the files that are loaded
        :return: A tuple containing the samples of the dataset and an array contain their corresponding class labels
        """
        logger.info("Loading dataset from " + self.data_dir)
        filename_list = fnmatch.filter(os.listdir(self.data_dir), self.file_prefix+"."+self.file_extension)
        filename_list = natsort.natsorted(filename_list)
        self.samples = []
        for filename in filename_list:
            path = os.path.join(self.data_dir, filename)
            sample = pd.read_csv(path, header=None)
            if scale:
                scaled_sample = sample.sub(sample.min(axis=0))
                scaled_sample = scaled_sample.div(scaled_sample.max(axis=0))
                sample = scaled_sample
            if display:
                logger.info(filename + " rows=" + str(sample.shape[0]))
            self.samples.append(sample)
        self.n_inputs = len(self.samples[0].columns)

        target_file_name = os.path.join(self.data_dir, "tar_class.csv")
        targets = pd.read_csv(target_file_name, header=None)
        self.targets = np.array(targets.iloc[:, 0])

        if len(self.samples) != len(self.targets):
            raise ValueError("The number of targets must be equal to the number of samples")

        (unique, counts) = np.unique(self.targets, return_counts=True)
        self._class_labels = {unique[i]: counts[i] for i in range(len(unique))}
        logger.info("Dataset loaded successfully. Samples = "+str(len(filename_list)) + ", class labels = " +
                    str(self._class_labels))
        return self.samples, self.targets

    def split(self, test_size: float = 0.25) -> tuple[list[pandas.DataFrame], list[pandas.DataFrame], np.ndarray,
                                                      np.ndarray]:
        """
        Splits the dataset into training and testing datasets. See the following cites related to training and testing
        datasets.
        https://stats.stackexchange.com/questions/152907/how-do-you-use-the-test-dataset-after-cross-validation
        https://machinelearningmastery.com/difference-test-validation-datasets/

        :param test_size: Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the
            test split. If None, the value is set to 0.25
        :return: A tuple having the training and testing datasets amd the training and testing targets
        """
        logger.info("Splitting data into training and testing datasets")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        train_index = []
        test_index = []
        for train_index, test_index in sss.split(self.samples, self.targets):
            logger.info("Train indexes(%s): %s Test indexes(%s): %s", str(len(train_index)),
                        np.array2string(train_index),
                        str(len(test_index)), np.array2string(test_index))
        training_data = [self.samples[i] for i in train_index]
        testing_data = [self.samples[i] for i in test_index]
        training_targets, testing_targets = np.array(self.targets[train_index]), np.array(self.targets[test_index])
        logger.info("Split completed. Training = %s samples, Testing = %s samples", str(len(training_data)),
                    str(len(testing_data)))
        return training_data, testing_data, training_targets, testing_targets

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, data_dir):
        self._data_dir = data_dir

    @property
    def file_prefix(self):
        return self._file_prefix

    @file_prefix.setter
    def file_prefix(self, file_prefix):
        self._file_prefix = file_prefix

    @property
    def targets_file_name(self):
        return self._targets_file_name

    @targets_file_name.setter
    def targets_file_name(self, targets_file_name):
        self._targets_file_name = targets_file_name

    @property
    def file_extension(self):
        return self._file_extension

    @file_extension.setter
    def file_extension(self, file_extension):
        self._file_extension = file_extension

    @property
    def n_inputs(self):
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, n_inputs):
        self._n_inputs = n_inputs

    @property
    def samples(self) -> list[pandas.DataFrame]:
        return self._samples

    @samples.setter
    def samples(self, samples: list[pandas.DataFrame]):
        self._samples = samples

    @property
    def targets(self) -> np.ndarray:
        return self._targets

    @targets.setter
    def targets(self, targets: np.ndarray):
        self._targets = targets

    def get_n_inputs(self) -> int:
        return self._n_inputs

    def set_n_inputs(self, n_inputs: int):
        self._n_inputs = n_inputs

    def get_samples(self) -> list[pandas.DataFrame]:
        return self._samples

    def set_samples(self, samples: list[pandas.DataFrame]):
        self._samples = samples

    def get_targets(self) -> np.ndarray:
        return self._targets

    def set_targets(self, targets: np.ndarray):
        self._targets - targets

    def get_class_labels(self) -> dict[int, int]:
        return self._class_labels
