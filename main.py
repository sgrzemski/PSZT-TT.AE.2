from GA import AlgorytmGenetyczny

import matplotlib.pyplot as plt


def main():
    seed = "PSZT"

    alg = AlgorytmGenetyczny(target=700, wielkosc_populacji=100,
                             dlugosc_indywidua=10, seed=seed)

    alg.run(100, verbose=True)

    print("\n")
    print("Rozwiazanie: \n {}".format(alg.best[0]))
    print("Loss: {}".format(alg.best[1]))

    # poka wykres
    plt.title('Historia ewolucji')
    plt.plot(alg.fitness_history, '-o', label='train')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.show()

if __name__ == '__main__':
    main()
