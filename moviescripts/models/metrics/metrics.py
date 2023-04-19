import numpy as np


class Accuracy:
    """Computes the overall accuracy of a classification model.

    The accuracy is computed as follows:

        accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

    Keyword arguments:
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the accuracy. Can be an int, or any iterable of ints.

    """

    def __init__(self):
        super().__init__()

    def value(self, conf_matrix):
        """Computes the overall accuracy.

        Returns:
            Float: The overall accuracy.
        """
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive
        true_negative = np.sum(conf_matrix) - (true_positive + false_positive + false_negative)

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

        return accuracy
