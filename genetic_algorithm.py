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

    def check(self, gene):
        """
        Validates whether given 'gene' passes the 3-SAT equation.

        Args:
            gene (Gene): The Gene to validate.

        Returns:
            (int, bool): Tuple of number of clauses passed and whether all clauses passed.
        """
        passed_clauses = 0
        for conjunctive in self._conjunctives:
            for v in conjunctive:
                if v < 0:
                    if not gene[abs(v) - 1]:
                        passed_clauses += 1
                        break
                else:
                    if gene[v - 1]:
                        passed_clauses += 1
                        break

        return passed_clauses, passed_clauses == self._clauses


class Gene(object):
    """Object representation of gene."""

    def __init__(self, equation: Equation, chromosomes: str):
        """Initializes object instance.

        Args:
            chromosomes (str): Chromosomes of gene.
            equation (Equation): 3-SAT equation gene is compared against.
        """
        self._equation = equation
        self._chromosomes = chromosomes
        self._valid = None
        self._fitness = None

    def __getitem__(self, index: int):
        """
        Gets chromosome at index.

        Args:
            index (int): Chromosome index.

        Returns:
            (Gene): Chromosome at index.
        """
        return self._chromosomes[index]

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

    def __setitem__(self, key: int, value: int):
        """
        Sets chromosome at index.

        Args:
            key (int): Index to insert chromosome.
            value (Gene): Value of chromosome.
        """
        self._chromosomes[key] = value
        self._fitness = None
        self._valid = None

    @property
    def chromosomes(self):
        """
        Gets chromosomes property.

        Returns:
            (str): Chromosomes.
        """
        return self._chromosomes

    @chromosomes.setter
    def chromosomes(self, value: str):
        """
        Sets chromosomes property of object.

        Args:
            value (str): New chromosomes property.
        """
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
        if self._fitness is None:
            clauses, passed = self._equation.check(self)
            self._fitness = clauses / self._equation.clauses
            self._valid = passed
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
            clauses, passed = self._equation.check(self)
            self._fitness = clauses / self._equation.clauses
            self._valid = passed
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
            value (Gene): Value of gene.
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
            (list): Genes of population.
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
            chromosomes = ''.join([str(random.choice([0, 1])) for _ in range(self._equation.variables)])
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
            parent1 (Gene): First parent gene.
            parent2 (Gene): Second parent gene.

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
