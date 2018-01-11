from GA import AlgorytmGenetyczny, nominaly, np
from operator import add
from functools import reduce
import math
import matplotlib.pyplot as plt

def count_not_zero_in_list(l):
       return reduce(add, list(map(lambda x: x>0, l)), 0)

def main():
    #seed = "PSZT"
    seed = None
    wielkosc_populacji=100
    zostaw= 0.2
    zostaw_losowo = 0.05
    mutuj = 0.01
    metoda_crossowania = 2 #1 - pół na pół, 2 na przemiennie
    kara_param = 100
    kara_param2 = 1.5
    vanilla=False
    iteracje = 20
    verbose = True
    # f-cja fitnes deluxe - nowe pomysly
    target=200


    alg = AlgorytmGenetyczny(target, wielkosc_populacji, seed, vanilla, zostaw,
                             zostaw_losowo, mutuj, metoda_crossowania,
                             kara_param, kara_param2)
    alg.run(iteracje, verbose)



    file = open("wyniki.log", mode='a+')
    file.write("\nNominaly {}".format(nominaly))
    file.write("\nCel: {}\n".format(target))
    file.write("Rozwiazanie:  {}\n".format(alg.best[0]))
    file.write("Liczba monet: {}\n".format(sum(alg.best[0])))
    file.write("Suma monet: {}\n".format(reduce(add, np.multiply(alg.best[0],nominaly))))
    file.write("Loss: {}\n".format(alg.best[1]))
    file.close()

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
