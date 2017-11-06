import random
import itertools
import math


class WOC(object):
    """Object representation of genetic algorithm."""

    def __init__(self, chromosomes: list, threshold: float):
        """
        Initializes Object Instance.

        Args:
            chromosomes (list): The chromosomes to perform the algorithm on.
            threshold (float): The aggregate threshold to use to choose which percentage of best genes to keep.
        """
        self.chromosomes = chromosomes
        self.threshold = threshold
        self.result = None

    def aggregate(self):
        # TODO: Implement this.
        pass
