from matplotlib import pyplot as plt
import numpy as np


def histogram(generations: list, **kwargs):
    """
    Display GA Histogram of single run.

    Args:
        generations (list): GA run generation's fittest fitnesses.
    """
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    # Generations
    x = list(range(len(generations)))
    y = generations

    ax.plot(x, y, color='black')

    if 'woc' in kwargs:
        ax.plot([0, len(args[0])], [kwargs['woc'], kwargs['woc']], color='blue')

    plt.xlim([0, len(generations)])
    ymin = round(min(generations), 1)
    plt.ylim([ymin, 1])
    plt.yticks(np.arange(round(min(y), 1), 1 + 0.01, 0.01))

    # Labels
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('GA Fittest Histogram')


def histograms(*args, **kwargs):
    """
    Display GA Histogram of multiple runs.

    Args:
        *args (args): Arguments should be lists containing generations fittest fitnesses.
        **kwargs (kwargs): Single argument woc is accepted.
    """
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    for generations in args:
        # Generations
        x = list(range(len(generations)))
        y = generations

        ax.plot(x, y, color='black')

    if 'woc' in kwargs:
        ax.plot([0, len(args[0])], [kwargs['woc'], kwargs['woc']], color='blue')

    plt.xlim([0, len(args[0])])
    ymin = round(min(g[0] for g in args), 1)
    plt.ylim([ymin, 1])
    plt.yticks(np.arange(ymin, 1 + 0.01, 0.01))

    # Labels
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('GA Fittests Histogram')


def show():
    plt.show()
