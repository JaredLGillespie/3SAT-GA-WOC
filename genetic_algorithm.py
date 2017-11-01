from math import *
import itertools
import random


class Equation(object):
    """Object representation of equation."""
    # TODO: Create equation object structure to be passed along each Gene

    def __init__(self, equation: str):
        """
        Initializes object instance.

        Args:
            equation (str): String representation of equation.
        """
        self._equation = equation

    def _repr_(self):
        return self._equation

    @property
    def variables(self):
        # TODO: Implement variables.
        raise NotImplementedError('TODO: Implement this!')


class Gene(object):
    """Object representation of gene."""

    def __init__(self, equation: Equation, chromosomes: str):
        """Initializes object instance.

        Args:
            chromosomes (str): Chromosomes of gene.
            equation (Equation): 3-SAT equation gene is compared against.
        """
        self._chromosomes = chromosomes
        self._valid = None
        self._fitness = None
        self._equation = None

    def __len__(self) -> int:
        """
        Length of object.

        Returns:
            (int): Length of chromosomes.
        """
        return len(self._chromosomes)

    def __repr__(self) -> str:
        """
        String representation of object.

        Returns:
            (str): String representation of object.
        """
        return self.chromosomes

    @property
    def chromosomes(self):
        return self._chromosomes

    @chromosomes.setter
    def chromosomes(self, value: str):
        self._chromosomes = value
        self._fitness = None
        self._valid = None

    @property
    def fitness(self):
        """
        Calculates fitness of gene.

        Returns:
            (float): Fitness of gene.
        """
        if not self.fitness is None:
            # TODO: Create fitness function and assign to self._fitness
            pass
        return self._fitness

    @property
    def valid(self) -> bool:
        """
        Returns 'true' if gene is valid, else 'false'.

        Gene is valid if passes 3-SAT equation..

        Returns:
            (bool): Validity of gene.
        """
        if self._valid is None:
            # TODO: Create validity function and assign to self._valid
            pass
        return self._valid

    def copy(self):
        """
        Creates and returns copy of gene object.

        Returns:
            (Gene): Returns copy of gene.
        """
        return Gene(self._equation, self._chromosomes)


class Population(object):
    """Object representation of population."""

    def __init__(self, size:int, equation: Equation):
        """
        Initializes object instance.

        Args:
            size (int): Population size.
            equation (Equation): Equation object.
        """
        self._size = size
        self._equation = equation
        self._genes = []
        self._fittest = None

    def __getitem__(self, index: int):
        """
        Gets gene at index.

        Args:
            index (int): Gene index.

        Returns:
            (Gene): Gene at index.
        """
        return self._genes[index]

    def __len__(self) -> int:
        """
        Current size of population.

        Returns:
            (int): Current size of population.
        """
        return len(self._genes)

    def __setitem__(self, key: int, value: Gene):
        """
        Sets gene at index.

        Args:
            key (int): Index to insert gene.
            value (Gene): Route to gene.
        """
        self._genes[key] = value
        self._fittest = None

    @property
    def equation(self) -> Equation:
        """
        SAT equation being passed around.

        Returns:
            (Equation): SAT equation.
        """
        return self._equation

    @property
    def fittest(self):
        """
        Gets the fittest gene.

        Returns:
            (Gene): The most fit gene.
        """
        if self._fittest is None:
            for gene in self._genes:
                if self._fittest is None:
                    self._fittest = gene
                else:
                    # If two most fit gene with the same fitness occur, the one which comes first is selected...
                    if self._fittest.fitness < gene.fitness:
                        self._fittest = gene
        return self._fittest

    @property
    def genes(self) -> list:
        """
        Genes of population.

        Returns:
            (list): Routes of population.
        """
        return self._genes[:]

    @property
    def size(self) -> int:
        """
        Size of population.

        Returns:
            (int): Size of population.
        """
        return self._size

    def add(self, gene: Gene):
        """
        Adds a gene to the population.

        Args:
            gene (Gene): The gene to add.
        """
        self._genes.append(gene)
        self._fittest = None

    def initialize(self):
        """Initializes population by generating random genes."""
        for i in range(self.size):
            chromosomes = ''.join([random.choice([0, 1]) for _ in range(len(self._equation.variables))])
            self.add(Gene(self._equation, chromosomes))


class GA(object):
    """Object representation of genetic algorithm."""

    def __init__(self, crossover_rate: float, mutation_rate: float):
        """
        Initializes Object Instance.

        Args:
            crossover_rate (float): The crossover rate.
            mutation_rate (float): The mutation rate.
        """
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate

    def evolve(self, population: Population) -> Population:
        """
        Evolves a population.

        Args:
            population (Population): Population to evolve.

        Returns:
            (Population): New Population.
        """
        new_population = Population(population.size, population.equation)

        # Fitness calculation
        fitnesses = [r.fitness for r in population]

        # Keep generating until new population size is adequate
        # Attempt to generate entire new population using crossovers from prior
        for _ in range(len(population)):
            parent1 = population[self.select(fitnesses)]
            parent2 = population[self.select(fitnesses)]

            # Perform crossover and add offspring to population
            if random.random() <= self._crossover_rate:
                child = self.crossover(parent1, parent2)

                if random.random() <= self._mutation_rate:
                    child = self.mutate(child)

                new_population.add(child)

        # Add fittest gene from old population to new population to keep the new size equal to old
        pop_len_diff = len(population) - len(new_population)

        if pop_len_diff > 0:
            pop_points_all: list = population.genes[:]
            pop_points_all.sort(key=lambda x: x.fitness, reverse=True)
            pop_points_fit = pop_points_all[:pop_len_diff]

            for point in pop_points_fit:
                # Perform mutations
                child = point
                if random.random() <= self._mutation_rate:
                    child = self.mutate(child)
                new_population.add(child)

        return new_population

    @staticmethod
    def crossover(self, parent1: Gene, parent2: Gene):
        """
        Performs a crossover between two parents, returning a child.

        Args:
            parent1 (Route): First parent gene.
            parent2 (Route): Second parent gene.

        Returns:
            (Gene): Child gene.
        """
        # TODO: Implement crossover function.
        return None

    @staticmethod
    def select(fitnesses: list) -> int:
        """
        Selects a parent index for crossover.

        Args:
            fitnesses (list): List of fitnesses of population to use in selection.

        Returns:
            (int): Parent index.
        """
        fitness = random.random() * sum(fitnesses)
        index = 0

        while fitness > 0:
            fitness -= fitnesses[index]
            index += 1

        if index < 0:
            index = 0

        return index - 1 if index > 0 else index

    @staticmethod
    def mutate(gene: Gene):
        """
        Performs a mutation of a gene.

        Args:
            gene (Gene): Gene to mutate.

        Returns:
            gene (Gene): Mutated gene.
        """
        # TODO: Implement mutation function.
        return None
