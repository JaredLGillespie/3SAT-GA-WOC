import random
import itertools
import math
from genetic_algorithm import Equation, Chromosome

class WOC(object):
    """Object representation of genetic algorithm."""

    def __init__(self, chromosomes: list, threshold: float, equation: Equation):
        """
        Initializes Object Instance.

        Args:
            chromosomes (list): The chromosomes to perform the algorithm on.
            threshold (float): The aggregate threshold to use to choose which percentage of best genes to keep.
        """
        self.chromosomes = chromosomes
        self.equation = equation
        self.threshold = threshold
        self.result = None
        self.counts = []


    def aggregate(self):
        #initalize counters
        for i in enumerate(self.chromosomes[0].genes):
            self.counts.append(0)
        #filter chromosomes based on threshold
        bestChromos = [x for x in self.chromosomes if x.fitness > self.threshold]
        #count for selecting wiseman
        for chromo in bestChromos:
            for i, gene in enumerate(chromo.genes):
                parity = 1
                if gene == 0:
                    parity = -1
                #add/subtract to counter in a weighted fashion
                self.counts[i] = self.counts[i] + chromo.fitness*parity
        #create wiseman
        wiseGenes = []
        for count in self.counts:
            gene = 1
            if(count <= 0):
                gene = 0
            wiseGenes.append(gene)
        wiseChromo = Chromosome(self.equation, wiseGenes)

        self.result = wiseChromo
