import random
import itertools
import math


class WOC(object):
    """Object representation of genetic algorithm."""

    def __init__(self, genes: list, threshold: float):
        """
        Initializes Object Instance.

        Args:
            genes (list): The genes to perform the algorithm on.
            threshold (float): The aggregate threshold to keep from the best chromosomes.
        """
        self.genes = genes
        self.threshold = threshold
        self.result = None

    def aggregate(self):
        # TODO: Implement this.
        pass
