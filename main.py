from GA import GeneticAlgorythm
import solver
from operator import add
from functools import reduce
import math
import numpy as np
import matplotlib.pyplot as plt
import timeit
import time


def test_algorithm(target, algorythm, iterations=10, verbose=False):
    ''' Z próby 'iterations wyznacza dokładność oraz średni czas obliczeń
    TODO: params, return
    '''
    times = []
    results = []
    for i in range(iterations):
        start = time.clock()
        results.append(algorythm() == target)
        end = time.clock()
        times.append(end - start)

    mean_time = np.mean(times)
    accuracy = np.sum(results) / len(results)

    if verbose:
        print("\t accuracy: {0}    mean time: {1:.8f}s"
              .format(accuracy, mean_time))

    return accuracy, time


def test_algorithm(target, algorythm, iterations=10, verbose=False):
    ''' Z próby 'iterations wyznacza dokładność oraz średni czas obliczeń
    TODO: params, return
    '''
    times = []
    corrects = []
    for i in range(iterations):
        start = time.clock()
        corrects.append(np.sum(algorythm()) == target)
        end = time.clock()
        times.append(end - start)

    mean_time = np.mean(times)
    accuracy = np.sum(corrects) / len(corrects)

    if verbose:
        print("\t accuracy: {0}    mean time: {1:.8f}s"
              .format(accuracy, mean_time))

    return accuracy, mean_time


def main():
    seed = 'PSZT'
    verbose = True

    params = dict()

    # Metody corssowania:
    # 1: pół na pół; 2: naprzemiennie
    params = {
        'population_size': 50,
        'elite_num': 2,
        'leave': 0.20,  # próg selekcji progowej
        'random_leave': 0.1,  # p-nstwo udzialu punktu w dalszej reprodukcji
        'mutation': 0.5,
        'crossing_method': 2
    }

    loss_params = {
        'penalty_param1': 100,
        'penalty_param2': 1.5
    }

    coins = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    target = 420

    params['elite_num'] = 2
    alg = GeneticAlgorythm(target, coins, **params, **loss_params, seed=seed, verbose=False)

    correct_solution = solver.coin_change_greedy(target, coins)
    result = alg.run(100)

    print()
    print("==========settings=============")
    print("target: {} \ncoins: {}".format(target, coins))
    print()
    print("----------results-------------")
    print("sum: ", np.sum(np.multiply(result[0], coins)))
    print("coin num: ", np.sum(result[0]))
    print("min loss: ", result[1])
    print()
    print("correct: ", correct_solution)
    print("our: ", result[0])

    # file = open("wyniki.log", mode='a+')
    # file.write("\nNominaly {}".format(nominaly))
    # file.write("\nCel: {}\n".format(target))
    # file.write("Rozwiazanie:  {}\n".format(alg.best[0]))
    # file.write("Liczba monet: {}\n".format(sum(alg.best[0])))
    # file.write("Suma monet: {}\n".format(reduce(add, np.multiply(alg.best[0],nominaly))))
    # file.write("Loss: {}\n".format(alg.best[1]))
    # file.close()

    # poka wykres
    # plt.title('Historia ewolucji')
    # plt.plot(alg.fitness_history, '-o', label='train')
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    #
    # plt.show()

if __name__ == '__main__':
    main()
