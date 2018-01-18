from GA import GeneticAlgorythm
import GA
import solver
import numpy as np

import matplotlib.pyplot as plt

import time
import random

def test_algorithm(target, algorythm, iterations=10, verbose=False, coins=None,
                   GA=False):
    ''' Z próby 'iterations wyznacza dokładność oraz średni czas obliczeń
    '''
    times = []
    corrects = []
    for i in range(iterations):
        start = time.clock()
        if GA:
            corrects.append(np.sum(np.multiply(our_solution[0], coins)) == target)

        else:
            corrects.append(np.sum(algorythm()) == target)

        end = time.clock()
        times.append(end-start)

    mean_time = np.mean(times)
    accuracy = np.sum(corrects) / len(corrects)

    if verbose:
        print("\t accuracy: {0}    mean time: {1:.8f}s"
              .format(accuracy, mean_time))

    return accuracy, mean_time



def main():

    # nasze standardowe nastawy algorytmu
    params = dict()
    # Metody corssowania:
    # 1: pół na pół; 2: naprzemiennie
    params = {
        'population_size': 100,
        'elite_num': 2,
        'leave': 0.2,  # próg selekcji progowej
        'random_leave': 0.05,  # p-nstwo udzialu punktu w dalszej reprodukcji
        'mutation': 0.5,
        'crossing_method': 2
    }
    loss_params = {
        'penalty_param1': 100,
        'penalty_param2': 1.5
    }


    coins=[1,2,5,10,20,50,100,200,500]
    seed="PSZT0"
    target = 1
    alg = GeneticAlgorythm(target, coins, **params, **loss_params, seed=seed,
                            verbose=False)


    # Tutaj zmieniałem parametry do przetestowania

    #  nazwa pliku z wykresem
    name='penalty'  # nazwa
    file_ext = 'png'  # typ pliku, do latexa dac 'eps'
    param_name = 'penalty'  # nazwa parametru widoczna na wykresie

    # wartoci paremetru do walidacji
    parameters = [1,2,5,10]

    show_plots = True
    seaborn_available = True  # dodatkowa paczka do wykresów
    if seaborn_available:
        import seaborn as sns

    # liczba iteracji w pojedyńczym uruchomieniu algorytmu
    alg_iterations = 200
    # liczba uruchomień algorytmu z różnymi wartociami seed
    iterations = 50



    bests = dict()
    best_history = dict()
    worst_history = dict()
    mean_loss = dict()
    best_log = dict()
    worst_log = dict()
    seeds = []

    print("==========settings=============")
    print("target: {} \ncoins: {}".format(target, coins))
    print("iterations: {}".format(iterations))
    print()
    for param in parameters:

        #########################
        # Zmiana parametru
        alg.penalty_param2 = param

        #########################

        bests[param] = []
        seeds = []

        for i in range(iterations):
            # change target
            alg.target = random.randint(1, 1000);
            # change seed for random generator
            seeds.append("PSZT{}".format(i))
            GA.random.seed(a="PSZT{}".format(i))
            random.seed(a="PSZT{}".format(i))  # just in case

            result = alg.run(alg_iterations)

            bests[param].append(alg.best[1])
            best_history[(param, i)] = alg.best_history

            worst_log[(param, i)] = alg.worst_log
            best_log[(param, i)] = alg.best_log
            mean_loss[(param, i)] = alg.fitness_history

            print("ITERATION {}".format(i))
            print("----------results-------------")
            print("sum: ", np.sum(np.multiply(result[0], coins)))
            print("coin num: ", np.sum(result[0]))
            print("loss: ", result[1])
            print()
            alg.annihilate_popultaion()

    # Wykres 1
    i=0

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_title("Najlepsze osobniki w populacji")

    for x in range(len(parameters)):
        axarr[0].plot(best_log[(parameters[x],i)], label=parameters[x])
    axarr[0].set_ylim(0, 500)
    axarr[1].set_title("Średnia populacji")

    ymax = 0;
    for x in range(len(parameters)):
        if ymax < max(mean_loss[(parameters[x],i)]):
            ymax = max(mean_loss[(parameters[x],i)])
        axarr[1].plot(mean_loss[(parameters[x],i)], label=parameters[x])
    axarr[1].set_ylim(0, ymax/2)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc=1)

    plt.savefig('./img/{param}_{typ}_{seed}.{format}'.format(param=name,typ='bmw',
                                                             seed=seeds[i],
                                                             format=file_ext),
                                                             format=file_ext,
                                                             dpi=1000)
    if show_plots:
        plt.show()


    # Wykres 2
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col',
                                               sharey='row',figsize=(15,10))

    ax1.set_title('{}={}'.format(param_name, parameters[0]))
    ax2.set_title('{}={}'.format(param_name, parameters[1]))
    ax3.set_title('{}={}'.format(param_name, parameters[2]))
    ax4.set_title('{}={}'.format(param_name, parameters[3]))
    ax1.set_ylim([0, 300])
    ax2.set_ylim([0, 300])
    ax3.set_ylim([0, 300])
    ax4.set_ylim([0, 300])

    for i in range(iterations):
        ax1.plot(best_history[(parameters[0],i)])
        ax2.plot(best_history[(parameters[1],i)])
        ax3.plot(best_history[(parameters[2],i)])
        ax4.plot(best_history[(parameters[3],i)])

    plt.savefig('./img/{param}_{typ}_{seed}.{format}'.format(param=name,typ='all',
                                                        seed="NA",
                                                        format=file_ext),
                                                        format=file_ext,
                                                        dpi=1000)
    if show_plots:
        plt.show()


    if seaborn_available:
        x1 = np.array(bests[parameters[0]])
        x2 = np.array(bests[parameters[1]])
        x3 = np.array(bests[parameters[2]])
        x4 = np.array(bests[parameters[3]])

        plt.figure(figsize=(15,10))
        X = [x1,x2,x3,x4]


        labels = [
            "{}".format(parameters[0]),
            "{}".format(parameters[1]),
            "{}".format(parameters[2]),
            "{}".format(parameters[3]),
        ]


        for x, label in zip(X,labels):
            sns.distplot(x, hist=True, rug=True,
                        hist_kws=dict(cumulative=True),
                        kde_kws=dict(cumulative=True),
                        label=label,
                        bins=50)

        plt.xlabel('F-cja celu')
        plt.ylabel('Znormalizowana częstość')
        plt.legend()
        plt.savefig('./img/{param}_{typ}_{seed}.{format}'.format(param=name,
                                                                typ='hist',seed="NA",
                                                                format=file_ext),
                                                                format=file_ext,
                                                                dpi=1000)

        if show_plots:
            plt.show()


if __name__ == '__main__':
    start = time.clock()
    main()
    end = time.clock()
    print("Time: ", end-start)