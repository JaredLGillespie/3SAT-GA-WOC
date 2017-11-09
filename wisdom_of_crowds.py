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
        # Initialize counters
        self.counts = list(0 for _ in range(len(self.chromosomes[0])))

        # Filter chromosomes based on threshold
        best_chromos = [x for x in self.chromosomes if x.fitness > self.threshold]

        # Count for selecting wiseman
        for chromo in best_chromos:
            for i, gene in enumerate(chromo.genes):
                parity = 1 if gene else -1
                # add/subtract to counter in a weighted fashion
                self.counts[i] += chromo.fitness * parity

        # Create wiseman
        wise_genes = []
        for count in self.counts:
            gene = 0 if count <= 0 else 1
            wise_genes.append(gene)

        self.result = Chromosome(self.equation, wise_genes)

