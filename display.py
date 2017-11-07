import matplotlib.pyplot as plt

# TODO: Add some of these bad boy graphs.

class GenImprovModel(object):
    def __init__(self, generation, fitness):
        self.generation = generation
        self.fitness = fitness

def plot_improvement(data: list):
    #plt.plot([1,2,3], [7,5,10]) #x,y
    for i in range(0, len(data)):
        plt.plot(data[i].generation, data[i].fitness, 'o')

    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.show()


#test = []
#test.append(GenImprovModel(4, 112))
#test.append(GenImprovModel(7, 93))
#test.append(GenImprovModel(11, 89))
#plot_improvement(test)
