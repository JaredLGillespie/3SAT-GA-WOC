import itertools
import math
import random


class Equation(object):
    """Object representation of equation."""

    def __init__(self, variables: int, clauses: int, conjunctives: list):
        """
        Initializes object instance.

        Args:
            variables (int): Number of variables.
            clauses (int): Number of clauses.
            conjunctives (list): Conjunctives of 3-SAT.
                Should be of form [(i, j, k), ...] where i, j, k are positive or negative integers representing
                a variable.
        """
        self._variables = variables
        self._clauses = clauses
        self._conjunctives = conjunctives
        self._equation = self.__build_equation()

    def __repr__(self):
        """
        String representation of object.

        Returns:
            (str): String representation of object.
        """
        return self._equation

    def __build_equation(self):
        """
        Builds the equation string representation.

        Returns:
            (str): String representation of equation.
        """
        e = []
        for i, conj in enumerate(self._conjunctives):
            e.append('(')
            for j, v in enumerate(conj):
                if v < 0:
                    e.append('¬' + str(abs(v)))
                else:
                    e.append(str(v))

                if j != len(conj) - 1:
                    e.append('∨')

            e.append(')')
            if i != len(self._conjunctives) - 1:
                e.append('∧')
        return ''.join(e)

    @property
    def clauses(self) -> int:
        """
        Gets clauses property of object.

        Returns:
            (int): Clauses.
        """
        return self._clauses

    @property
    def variables(self) -> int:
        """
        Gets variables property of object.

        Returns:
            (int): Variables.
        """
        return self._variables

    def check(self, chromosome):
        """
        Validates whether given 'chromosome' passes the 3-SAT equation.

        Args:
            chromosome (Chromosome): The Chromosome to validate.

        Returns:
            (int, bool): Tuple of number of clauses passed and whether all clauses passed.
        """
        passed_clauses = 0
        for conjunctive in self._conjunctives:
            for v in conjunctive:
                if v < 0:
                    if not chromosome[abs(v) - 1]:
                        passed_clauses += 1
                        break
                else:
                    if chromosome[v - 1]:
                        passed_clauses += 1
                        break

        return passed_clauses, passed_clauses == self._clauses


class Chromosome(object):
    """Object representation of chromosome."""

    def __init__(self, equation: Equation, genes: list):
        """Initializes object instance.

        Args:
            genes (list): Genes of chromosome.
            equation (Equation): 3-SAT equation chromosome is compared against.
        """
        self._equation = equation
        self._genes = genes
        self._valid = None
        self._fitness = None

    def __getitem__(self, index: int):
        """
        Gets genes at index.

        Args:
            index (int): Gene index.

        Returns:
            (Gene): Gene at index.
        """
        return self._genes[index]

    def __len__(self) -> int:
        """
        Length of object.

        Returns:
            (int): Number of genes in chromosome.
        """
        return len(self._genes)

    def __repr__(self) -> str:
        """
        String representation of object.

        Returns:
            (str): String representation of object.
        """
        return ''.join(map(str, self._genes))

    def __setitem__(self, key: int, value: int):
        """
        Sets chromosome at index.

        Args:
            key (int): Index to insert chromosome.
            value (Gene): Value of chromosome.
        """
        self._genes[key] = value
        self._fitness = None
        self._valid = None

    @property
    def equation(self) -> Equation:
        """
        Returns equation property of object.

        Returns:
            (Equation): The equation property of the object.
        """
        return self._equation

    @property
    def fitness(self):
        """
        Calculates fitness of chromosome.

        Returns:
            (float): Fitness of chromosome.
        """
        if self._fitness is None:
            clauses, passed = self._equation.check(self)
            self._fitness = clauses / self._equation.clauses
            self._valid = passed
        return self._fitness

    @property
    def genes(self):
        """
        Gets genes property.

        Returns:
            (str): Genes.
        """
        return self._genes

    @genes.setter
    def genes(self, value: str):
        """
        Sets genes property of object.

        Args:
            value (str): New genes property.
        """
        self._genes = value
        self._fitness = None
        self._valid = None

    @property
    def valid(self) -> bool:
        """
        Returns 'true' if chromosome is valid, else 'false'.

        Chromosome is valid if passes 3-SAT equation..

        Returns:
            (bool): Validity of chromosome.
        """
        if self._valid is None:
            clauses, passed = self._equation.check(self)
            self._fitness = clauses / self._equation.clauses
            self._valid = passed
        return self._valid

    def copy(self):
        """
        Creates and returns copy of chromosome object.

        Returns:
            (Chromosome): Returns copy of chromosome.
        """
        return Chromosome(self._equation, self._genes[:])


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
        self._chromosomes = []
        self._fittest = None

    def __getitem__(self, index: int) -> Chromosome:
        """
        Gets chromosome at index.

        Args:
            index (int): Chromosome index.

        Returns:
            (Chromosome): Chromosome at index.
        """
        return self._chromosomes[index]

    def __len__(self) -> int:
        """
        Current size of population.

        Returns:
            (int): Current size of population.
        """
        return len(self._chromosomes)

    def __repr__(self):
        """
        String representation of object.

        Returns:
            (str): String representation of object.
        """
        return str(self._chromosomes)

    def __setitem__(self, key: int, value: Chromosome):
        """
        Sets chromosome at index.

        Args:
            key (int): Index to insert chromosome.
            value (Chromosome): Value of chromosome.
        """
        self._chromosomes[key] = value
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
        Gets the fittest chromosome.

        Returns:
            (Chromosome): The most fit chromosome.
        """
        if self._fittest is None:
            for chromosome in self._chromosomes:
                if self._fittest is None:
                    self._fittest = chromosome
                else:
                    # If two most fit chromosomes with the same fitness occur, the one which comes first is selected...
                    if self._fittest.fitness < chromosome.fitness:
                        self._fittest = chromosome
        return self._fittest

    @property
    def chromosomes(self) -> list:
        """
        Chromosomes of population.

        Returns:
            (list): Chromosomes of population.
        """
        return self._chromosomes[:]

    @property
    def size(self) -> int:
        """
        Size of population.

        Returns:
            (int): Size of population.
        """
        return self._size

    def add(self, chromosome: Chromosome):
        """
        Adds a chromosome to the population.

        Args:
            chromosome (Chromosome): The chromosome to add.
        """
        self._chromosomes.append(chromosome)
        self._fittest = None

    def initialize(self):
        """Initializes population by generating random chromosomes."""
        for i in range(self.size):
            genes = [random.choice([0, 1]) for _ in range(self._equation.variables)]
            self.add(Chromosome(self._equation, genes))


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

        # Add fittest chromosome from old population to new population to keep the new size equal to old
        pop_len_diff = len(population) - len(new_population)

        if pop_len_diff > 0:
            pop_points_all: list = population.chromosomes[:]
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
    def crossover(parent1: Chromosome, parent2: Chromosome):
        """
        Performs a crossover between two parents, returning a child. Random index to split between left and right
        parent is chosen. Genes left of split from parent 1 and genes right of split from parent 2 are used to form
        new child.

        Args:
            parent1 (Chromosome): First parent chromosome.
            parent2 (Chromosome): Second parent chromosome.

        Returns:
            (Chromosome): Child chromosome.
        """
        split_index = -1
        if len(parent1) == 2:
            split_index = 1
        else:
            split_index = random.randint(0, len(parent1) - 2)

        genes = parent1[0:split_index] + parent2[split_index:]

        return Chromosome(parent1.equation, genes)

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
    def mutate(chromosome: Chromosome):
        """
        Performs a mutation of a chromosome. Randomly mutates a random number of genes in the chromosome.

        Args:
            chromosome (Chromosome): Chromosome to mutate.

        Returns:
            chromosome (Chromosome): Mutated chromosome.
        """
        genes = chromosome.genes[:]
        gene_choices = set(range(len(chromosome)))
        mutation_indexes = []

        # Mutate between 1 and all genes...
        for _ in range(random.randint(1, len(chromosome) - 1)):
            c = random.choice(list(gene_choices))
            mutation_indexes.append(c)
            gene_choices.remove(c)

        for i in mutation_indexes:
            genes[i] = int(not genes[i])

        return Chromosome(chromosome.equation, genes)
