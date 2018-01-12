from GA import AlgorytmGenetyczny, nominaly, np
import solver
from operator import add
from functools import reduce
import math
import matplotlib.pyplot as plt

import timeit
import time


def count_not_zero_in_list(l):
       return reduce(add, list(map(lambda x: x>0, l)), 0)


def test_GA(target, fn, iterations=10, verbose=False):
    ''' Z próby 'iterations wyznacza dokładność oraz średni czas obliczeń
    TODO: params, return
    '''
    times = []
    results = []
    for i in range(iterations):
        start = time.clock()
        results.append(fn() == target)
        end = time.clock()
        times.append(end-start)

    mean_time = np.mean(times)
    accuracy = np.sum(results) / len(results)

    if verbose:
        print("\t accuracy: {0}    mean time: {1:.8f}s"
              .format(accuracy, mean_time))

    return accuracy, time


def compare_algorithms(target, alg, iterations=10, verbose=True):
    ''' Porównuje czasowo algorytmy
    TODO: opis f-cji i komentarze
    '''
    coins = [1,2,5,10,20,50,100,200,500]
    algorythms = [alg, lambda: solver.coin_change_greedy(target, coins),
                  lambda: solver.coin_change_dp(target, coins)]
    algorythms_names = ['GA', 'Greedy', 'Dynamic programming']

    results = []
    timings = []

    for i, alg in enumerate(algorythms):
        results.append(alg())
        timings.append(timeit.timeit(alg, number=iterations))
        if verbose:
            print("{0}. {1}\t coins: {2} \t time: {3:.8f}"
                .format(i, algorythms_names[i], len(results[i]), timings[i]))

    return results, timings


def main():
    #seed = "PSZT"
    seed = None
    wielkosc_populacji=100
    zostaw= 0.2
    zostaw_losowo = 0.05
    mutuj = 0.01
    metoda_crossowania = 1 #1 - pół na pół, 2 na przemiennie
    kara_param = 100
    kara_param2 = 1.5
    vanilla=False
    iteracje = 100
    verbose = False
    # f-cja fitnes deluxe - nowe pomysly
    target=200


    alg = AlgorytmGenetyczny(target, wielkosc_populacji, seed, vanilla, zostaw,
                             zostaw_losowo, mutuj, metoda_crossowania,
                             kara_param, kara_param2, verbose)

    #alg.run(iteracje, verbose=False)


    test_GA(target, lambda: alg.run(iteracje), iterations=20, verbose=True)
    #compare_algorithms(target, lambda: alg.run(iteracje), 10)

    
    #file = open("wyniki.log", mode='a+')
    #file.write("\nNominaly {}".format(nominaly))
    #file.write("\nCel: {}\n".format(target))
    #file.write("Rozwiazanie:  {}\n".format(alg.best[0]))
    #file.write("Liczba monet: {}\n".format(sum(alg.best[0])))
    #file.write("Suma monet: {}\n".format(reduce(add, np.multiply(alg.best[0],nominaly))))
    #file.write("Loss: {}\n".format(alg.best[1]))
    #file.close()

    ## możliwie najprostrza f-cja fitness
    #alg_vanilla = AlgorytmGenetyczny(target=700, wielkosc_populacji=100,
                                     #dlugosc_indywidua=20, seed=seed, vanilla=True)
#
    #alg_vanilla.run(500, verbose=False)
#
    #print("\n Vanilla:")
    #print("Rozwiazanie: \n {}".format(alg_vanilla.best[0]))
    #print("Suma monet: {}".format(sum(alg_vanilla.best[0])))
    #print("Liczba monet: {}".format(count_not_zero_in_list(alg_vanilla.best[0])))
    #print("Loss: {}".format(alg_vanilla.best[1]))

    # poka wykres
    #plt.title('Historia ewolucji')
    #plt.plot(alg.fitness_history, '-o', label='train')
    #plt.xlabel('Iteration')
    #plt.ylabel('Loss')
#
    #plt.show()

if __name__ == '__main__':
    main()
