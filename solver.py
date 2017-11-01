import getopt
import sys
import timeit

import display
from genetic_algorithm import GA, Population, Equation
from wisdom_of_crowds import WOC


def solve(equation: str, mutation_rate: float, crossover_rate: float, population_size: int, generations: int,
          repetitions: int, threshold: float, is_verbose: bool, show_display: bool):
    """
    Solves a given 3-SAT 'equation' with the supplied parameters via GA and WOC.

    Args:
        equation (str): 3-SAT equation to solve.
        mutation_rate (float): Probability of mutation.
        crossover_rate (float): Probability of crossover.
        population_size (int): Size of population for genetic algorithm.
        generations (int): Number of generations for genetic algorithm.
        repetitions (int): Number of genetic algorithms to run.
        threshold (float): Threshold limit to use for wisdom of crowds algorithm.
        is_verbose (bool): Whether verbose output should be displayed.
        show_display (bool): Whether graphical output should be displayed.
    """
    f = format('{0:<20} {1}')

    # TODO: Break apart equation into some sort of gene accepting object.
    eq = Equation(equation)

    # Store the fittest of each GA run
    ga_results = []

    #region GA
    # TODO: Split into multiple threads
    for i in range(repetitions):
        print('GA %s of %s' % (i + 1, repetitions))

        # Start GA
        tic = timeit.default_timer()

        ga = GA(crossover_rate, mutation_rate)
        population = Population(population_size, eq)

        population.initialize()

        if is_verbose:
            print(f.format('Generation 0: ', population.fittest.distance))

        # Evolve population
        for g in range(generations):
            population = ga.evolve(population)
            if is_verbose:
                print(f.format('Generation %s: ' % (g + 1), population.fittest.distance))

            # TODO: Add GA graphs (in separate thread)

        toc = timeit.default_timer()
        # End GA algorithm

        if is_verbose:
            print(f.format('Fittest', population.fittest))
            print(f.format('Fittest Fitness.', population.fittest.fitness))

        print(f.format('Time (seconds)', toc - tic) + '\n')

        ga_results.append(population.fittest)
    #endregion GA

    #region WOC
    tic = timeit.default_timer()
    woc = WOC(ga_results, threshold)
    woc.aggregate()
    toc = timeit.default_timer()
    #endregion WOC

    ga_fitnesses = [g.fitness for g in ga_results]
    print(f.format('GA Min Dist.', min(ga_fitnesses)))
    print(f.format('GA Max Dist.', max(ga_fitnesses)))
    print(f.format('GA Avg Dist.', sum(ga_fitnesses) / len(ga_fitnesses)))
    print(f.format('WOC Dist.', woc.result))
    print(f.format('WOC Time (seconds)', toc - tic))

    # Display graphs
    if show_display:
        # TODO: Add WOC graphs (in separate thread)
        pass


def usage():
    """
    Displays command-line usage information.
    """
    f = format('{0:<40} {1}')
    r = list()
    r.append('Usage: solver -e sat -c crossover-rate -m mutation-rate -p population-size -g number-of-generations -r ' +
             'repetitions -a aggregate-limit [-v] [-d]')
    r.append('')
    r.append('Options:')
    r.append(f.format('\t-h --help', 'Display command usage.'))
    r.append(f.format('\t-e --equation sat', 'Boolean satisfiability problem to solve.'))
    r.append(f.format('\t-c --crossover crossover-rate', 'Crossover percentage rate between [0, 1].'))
    r.append(f.format('\t-m --mutation mutation-rate', 'Mutation percentage rate between [0, 1].'))
    r.append(f.format('\t-p --population population-size', 'Size of populations for genetic algorithm.'))
    r.append(f.format('\t-g --generation number-of-generations', 'Number of generations to use in genetic algorithm.'))
    r.append(f.format('\t-r --repeat repetitions', 'Number of genetic algorithm runs to aggregate for WOC.'))
    r.append(f.format('\t-a --aggregate aggregate-limit', 'Aggregate minimum threshold limit for WOC algorithm.'))
    r.append(f.format('\t-v --verbose', 'Display verbose output (time, statistics, etc.).'))
    r.append(f.format('\t-d --display', 'Display graphical output.'))
    r.append('')
    print('\n'.join(r))


if __name__ == '__main__':
    equation, crossover_rate, mutation_rate = None, None, None
    population_size, generations, repetitions, aggregate = None, None, None, None
    is_verbose, show_display = False, False

    # Get command line arguments
    try:
        shorthand_args = 'he:c:m:p:g:r:a:vd'
        longhand_args = ['help', 'equation=', 'crossover=', 'mutation=', 'population=', 'generations=', 'repetitions=',
                         'aggregate=', 'verbose', 'display']
        opts, args = getopt.getopt(sys.argv[1:], shorthand_args, longhand_args)
    except getopt.GetoptError as e:
        print(e)
        usage()
        sys.exit(2)

    # Interpret command line arguments
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        elif opt in ('-e', '--equation'):
            equation = arg
        elif opt in ('-c', '--crossover'):
            try:
                crossover_rate = float(arg)

                if crossover_rate < 0 or crossover_rate > 1:
                    raise ValueError('Crossover rate must be within [0, 1]!')
            except ValueError as e:
                print(e)
                usage()
                sys.exit(2)
        elif opt in ('-m', '--mutation'):
            try:
                mutation_rate = float(arg)

                if mutation_rate < 0 or mutation_rate > 1:
                    raise ValueError('Mutation rate must be within [0, 1]!')
            except ValueError as e:
                print(e)
                usage()
                sys.exit(2)
        elif opt in ('-p', '--population'):
            try:
                population_size = int(arg)

                if population_size < 1:
                    raise ValueError('Size of population must be > 0!')
            except ValueError as e:
                usage()
                sys.exit(2)
        elif opt in ('-g', '--generation'):
            try:
                generations = int(arg)

                if generations < 1:
                    raise ValueError('Number of generations must be > 0!')
            except ValueError as e:
                print(e)
                usage()
                sys.exit(2)
            generations = int(arg)
        elif opt in ('-r', '--repetitions'):
            try:
                repetitions = int(arg)

                if repetitions < 1:
                    raise ValueError('Repetitions must be >= 1!')
            except ValueError as e:
                print(e)
                usage()
                sys.exit(2)
        elif opt in ('-a', '--aggregate'):
            try:
                aggregate = float(arg)

                if aggregate < 0 or aggregate > 1:
                    raise ValueError('Aggregate threshold must be within [0, 1]!')
            except ValueError as e:
                print(e)
                usage()
                sys.exit(2)
        elif opt in ('-v', '--verbose'):
            is_verbose = True
        elif opt in ('-d', '--display'):
            show_display = True

    # Validate required parameters were given
    if not (equation and mutation_rate and crossover_rate and population_size and generations
            and repetitions and aggregate):
        print('Not all required parameter given!')
        usage()
        sys.exit(2)

    solve(equation,
          mutation_rate,
          crossover_rate,
          population_size,
          generations,
          repetitions,
          aggregate,
          is_verbose,
          show_display)
