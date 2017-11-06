import getopt
import math
import os
import sys
import timeit
from multiprocessing import Pool

import display
from genetic_algorithm import GA, Population, Equation
from wisdom_of_crowds import WOC


def load_equation(file_path: str):
    """
    Load and read file with SAT equation. File should be of the following format:
        header
        p cnf variables clauses
        vi vii viii
        vj vjj vjjj
        ...
    Utilize https://toughsat.appspot.com/ for correctly formatted problems in 3CNF format.

    Args:
        file_path (str): Path of file to read.

    Returns:
        Equation: Built 3CNF equation object.
    """
    variables, clauses = 0, 0
    cunjunctives = []
    with open(file_path, 'r') as file:
        i = 0
        for ln in file.read().split('\n'):
            i += 1
            # Skip first line
            if i == 1:
                continue
            elif i == 2:
                variables, clauses = tuple(map(int, ln.split()[2:]))
            elif ln:
                cunjunctives.append(list(map(int, ln.split()[:-1])))

    return Equation(variables, clauses, cunjunctives)


def run_ga(data: tuple):
    """
    Runs the genetic-algorithm. Thread-safe.

    Args:
        data (tuple): A tuple of the number of the iteration executing the function, the number of repetitions, the
                      equation to use for the genetic algorithm, the crossover rate, the mutation rate, the population
                      size, the number of generations, and whether to display graphs, and whether the output is verbose.
                      (i.e. (1, 10, Equation, 0.9, 0.001, 100, 1000, True, False))

    Returns:
        (Gene): Returns the fittest gene of the algorithm.
    """
    iteration_number, repetitions, equation, crossover_rate, mutation_rate, population_size, generations, show_display, is_verbose = data
    f = format('{0:<20} {1}')
    print('GA %s of %s' % (iteration_number, repetitions))

    # Start GA
    tic = timeit.default_timer()

    ga = GA(crossover_rate, mutation_rate)
    population = Population(population_size, equation)

    population.initialize()

    if is_verbose:
        print(f.format('(GA %s) Generation 0: ' % iteration_number, population.fittest.fitness))

    # Evolve population
    for g in range(generations):
        population = ga.evolve(population)
        if is_verbose:
            print(f.format('(GA %s) Generation %s: ' % (iteration_number, g + 1), population.fittest.fitness))

            # TODO: Add GA graphs (in separate thread)

    toc = timeit.default_timer()
    # End GA algorithm

    if is_verbose:
        print(f.format('(GA %s) Fittest' % iteration_number, population.fittest))
        print(f.format('(GA %s) Fittest Fitness.' % iteration_number, population.fittest.fitness))

    print(f.format('(GA %s) Time (seconds)' % iteration_number, toc - tic) + '\n')

    return toc - tic, population.fittest


def solve(equation: Equation, mutation_rate: float, crossover_rate: float, population_size: int, generations: int,
          repetitions: int, threshold: float, threads: int, is_verbose: bool, show_display: bool):
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
        threads (int): Number of threads to use in genetic algorithm.
        is_verbose (bool): Whether verbose output should be displayed.
        show_display (bool): Whether graphical output should be displayed.
    """
    f = format('{0:<20} {1}')

    # Used to populate thread function parameters
    def ga_it():
        for i in range(1, repetitions + 1):
            yield i, repetitions, equation, crossover_rate, mutation_rate, population_size, generations, show_display, is_verbose

    # Run GAs
    ga_results = []
    ga_times = []

    ga_toc = 0
    ga_tic = 0
    if threads == 0:
        for d in ga_it():
            gt, gr = run_ga(d)
            ga_results.append(gr)
            ga_times.append(gt)
    else:
        ga_tic = timeit.default_timer()
        pool = Pool(processes=threads)
        it = pool.map(run_ga, ga_it(), chunksize=1)
        ga_results = [i[1] for i in it]
        ga_times = [i[0] for i in it]
        ga_toc = timeit.default_timer()

    # Run WOC
    woc_tic = timeit.default_timer()
    woc = WOC(ga_results, threshold)
    woc.aggregate()
    woc_toc = timeit.default_timer()

    # Print final results
    ga_fitnesses = [g.fitness for g in ga_results]
    min_fit = min(ga_fitnesses)
    max_fit = max(ga_fitnesses)
    avg_fit = sum(ga_fitnesses) / len(ga_fitnesses)
    print(f.format('Equation', equation))
    print(f.format('GA Min Fitness', '%s [%s]' % (min_fit, [g for g in ga_results if g.fitness == min_fit][0])))
    print(f.format('GA Max Fitness',  '%s [%s]' % (max_fit, [g for g in ga_results if g.fitness == max_fit][0])))
    print(f.format('GA Avg Fitness', avg_fit))
    print(f.format('GA Min Time (seconds)', min(ga_times)))
    print(f.format('GA Max Time (seconds)', max(ga_times)))
    print(f.format('GA Avg Time (seconds)', sum(ga_times) / len(ga_times)))
    print(f.format('GA Total Time (seconds)', sum(ga_times)))
    print(f.format('GA Thread Time (seconds)', ga_toc - ga_tic))
    print(f.format('WOC Dist.', woc.result))
    print(f.format('WOC Time (seconds)', woc_toc - woc_tic))

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
             'repetitions -a aggregate-limit [-t thread-count] [-v] [-d]')
    r.append('')
    r.append('Options:')
    r.append(f.format('\t-h --help', 'Display command usage.'))
    r.append(f.format('\t-f --file sat-file', 'File with 3-CNF to solve.'))
    r.append(f.format('\t-c --crossover crossover-rate', 'Crossover percentage rate between [0, 1].'))
    r.append(f.format('\t-m --mutation mutation-rate', 'Mutation percentage rate between [0, 1].'))
    r.append(f.format('\t-p --population population-size', 'Size of populations for genetic algorithm.'))
    r.append(f.format('\t-g --generation number-of-generations', 'Number of generations to use in genetic algorithm.'))
    r.append(f.format('\t-r --repeat repetitions', 'Number of genetic algorithm runs to aggregate for WOC.'))
    r.append(f.format('\t-a --aggregate aggregate-limit', 'Aggregate minimum threshold limit for WOC algorithm.'))
    r.append(f.format('\t-t threads thread-count', 'Number of threads to use in the genetic-algorithm.'))
    r.append(f.format('\t-v --verbose', 'Display verbose output (time, statistics, etc.).'))
    r.append(f.format('\t-d --display', 'Display graphical output.'))
    r.append('')
    print('\n'.join(r))


if __name__ == '__main__':
    equation, crossover_rate, mutation_rate = None, None, None
    population_size, generations, repetitions, aggregate = None, None, None, None
    is_verbose, show_display = False, False
    threads = None

    # Get command line arguments
    try:
        shorthand_args = 'hf:c:m:p:g:r:a:t:vd'
        longhand_args = ['help', 'file=', 'crossover=', 'mutation=', 'population=', 'generations=', 'repetitions=',
                         'aggregate=', 'threads=', 'verbose', 'display']
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
        elif opt in ('-f', '--file'):
            if os.path.isabs(arg):
                file = arg
            else:
                file = os.path.join(os.getcwd(), arg)
            equation = load_equation(file)
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
        elif opt in ('-t', '--threads'):
            # If -t "thread_count" not given, defaults to cpu_count()
            try:
                threads = int(arg)

                if threads < 0:
                    raise ValueError('Threads must be >= 0!')
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
          threads,
          is_verbose,
          show_display)
