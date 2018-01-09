from GA import AlgorytmGenetyczny
from operator import add
from functools import reduce

import matplotlib.pyplot as plt


def count_not_zero_in_list(l):
       return reduce(add, list(map(lambda x: x>0, l)), 0)

def main():
    #seed = "PSZT"
    seed = None

    # f-cja fitnes deluxe - nowe pomysly
    alg = AlgorytmGenetyczny(target=52, wielkosc_populacji=10, seed=seed)

    alg.run(iteracje=10, verbose=False)

    print("\n New:")
    print("Rozwiazanie: \n {}".format(alg.best[0]))
    print("Suma monet: {}".format(sum(alg.best[0])))
    print("Liczba monet: {}".format(count_not_zero_in_list(alg.best[0])))
    print("Loss: {}".format(alg.best[1]))

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
    plt.title('Historia ewolucji')
    plt.plot(alg.fitness_history, '-o', label='train')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.show()

if __name__ == '__main__':
    main()
