import random
from operator import add
from functools import reduce
import math
import numpy as np
import datetime
import copy

#self.coins = [1, 2, 5, 10, 20, 50, 100, 200, 500] #Self.Coins PLN, tylko calkowite

class AlgorytmGenetyczny:
    def __init__(self, target, coins, wielkosc_populacji=100, seed=None, vanilla=False,
    zostaw=0.2, zostaw_losowo=0.05, mutuj=0.01, metoda_crossowania=1,
                 kara_param=1000, kara_param2=1.5, elite_num=2, verbose=False):
        '''
        Inicjacja algorytmu
        :param target: kwota do wydania
        :param wielkosc_populacji: wielkość generowanej co iteracje populacji
        :param dlugosc_indywidua: maksymalna liczba wydawanych monet
        '''
        random.seed(a=seed)

        self.target = target
        self.coins = coins
        self.wielkosc_populacji = wielkosc_populacji
        self.dlugosc_indywidua = len(self.coins)
        self.zostaw = zostaw
        self.zostaw_losowo = zostaw_losowo
        self.mutuj = mutuj
        self.metoda_crossowania = metoda_crossowania
        self.kara_param = kara_param
        self.kara_param2 = kara_param2

        # init elite
        self.elite_num = elite_num
        self.elite = [self.stworz_indywidual() for e in range(self.elite_num)]

        # init population
        self.population = self.stworz_populacje(self.wielkosc_populacji)
        self.population.extend(self.elite)

        self.fitness_history = []  # log of mean population fitness
        self.best_history = []  # log of lowest loss value


        self.worse = None
        self.best = None
        # for test sake
        self.vanilla = vanilla
        self.verbose = verbose


    def wylosuj_ilosc_sztuk_nominalu(self, poz):
        a = math.floor(self.target/self.coins[poz])
        if a != 0:
            a = random.randint(0, a)
        return a


    def stworz_indywidual(self):
        '''
        Funkcja do generowania indywiduow.
        :param dlugosc: okresla ilosc elementow w indywiduale
        :return: zwraca iteral zawierajacy self.coins PLN
        '''
        list = []
        for x in range(self.dlugosc_indywidua) :
            list.append(self.wylosuj_ilosc_sztuk_nominalu(x))
        return list


    def stworz_populacje(self, ilesztuk):
        '''
        Funkcja do tworzenia populacji z indywiduow.
        :param ilesztuk: ilosc generowanych indywiduow w populacji
        :return: zwraca populacje zlozana z iteralow
        '''
        return [self.stworz_indywidual() for x in range(ilesztuk)]


    def fitness(self, indywidual, cel):
        '''
        Funkcja sprawdzajaca bliskosc iteralu od zalozonego celu/reszty.
        :param indywidual: sprawdzane indywiduum
        :param cel: zalozona wartosc celu
        :return: zwraca wartosc bezwgledna sumy iteralu i celu
        '''
        # TODO: cross validate


        suma =  reduce(add, np.multiply(indywidual,self.coins))

        loss =abs(cel - suma)

        liczba_monet = reduce(add, indywidual)

        #print(self.coins)
        #print(indywidual)
        #print("Suma " ,suma)
        #print("Liczba monet" ,liczba_monet)

        # kara za nie osiagniecie targetu
        kara = loss * self.kara_param

        if self.vanilla:
            liczba_monet = 0
            self.kara_param = 1000

        return self.kara_param * loss**2 + self.kara_param2*liczba_monet


    def jakosc(self, populacja, cel):
        '''
        Funkcja sprawdzajaca jakosc calej populacji za pomoca funkcji fitness.
        :param populacja: zsprawdzana populacja
        :param cel: zalozona wartosc celu
        :return: zwracana jest lista jakosci dla kazdego indywidualu
        '''
        summed = reduce(add, (self.fitness(x, cel) for x in populacja), 0)
        return summed / (len(populacja) * 1.0)


    def annihilate_popultaion(self):
        self.population = self.stworz_populacje(self.wielkosc_populacji)
        self.best_history = []
        self.fitness_history = []
        self.best = None
        self.worse = None
        self.elite = [self.stworz_indywidual() for e in range(self.elite_num)]



    def selection(self, population):
        '''Wybór osobników z populacji do reprodukcji

        Selekcja wybiera 20% najlepszych punktów oraz z pewnym
        prawdopodobieństem brane są punkty z pozostałej części populacji
        '''

        # tworzy liste jakosci dla kazdego indywiduum
        # na przyklad (180, [80, 75, 8, 85, 57])
        graded = [(self.fitness(x, self.target), x) for x in population]

        # posortowana lista jakosci z usunietym fitnesem
        graded = [x[1] for x in sorted(graded)]

        # ilosc indywiduow, ktore przetrwaja
        zostaw_dlugosc = int(len(graded) * self.zostaw)

        #wyznacza z listy jakosci najlepsze iteraly
        rodzice = graded[:zostaw_dlugosc]

        #losowo wybierz inne indywidyaly zeby zapewnic rownosc genetyczna
        #z listy odrzuconych wybierz iteral
        for individual in graded[zostaw_dlugosc:]:
            if self.zostaw_losowo > random.random():
                rodzice.append(individual)

        return rodzice


    def mutation(self, population):
        '''Mutacja losowych osobników z populacji'''
        for individual in population:
            if self.mutuj > random.random():
                pos_to_mutate = random.randint(0, len(individual) - 1)
                individual[pos_to_mutate] = self.wylosuj_ilosc_sztuk_nominalu(pos_to_mutate)

        return population


    def crossover(self, mating_pool):
        ''' Krzyżowanie osobników wybranych do reprodukcji
        '''

        # crossover parents to create children
        rodzice_dlugosc = len(mating_pool)
        rzadana_dlugosc = self.wielkosc_populacji  - rodzice_dlugosc

        dzieci = []
        while len(dzieci) < rzadana_dlugosc:
            meski = random.randint(0, rodzice_dlugosc - 1)
            damski = random.randint(0, rodzice_dlugosc - 1)
            if meski != damski:
                meski = mating_pool[meski]
                damski = mating_pool[damski]
                if(self.metoda_crossowania == 1):
                    polowa = int(len(meski) / 2)
                    dziecko = meski[:polowa] + damski[polowa:]
                    dzieci.append(dziecko)
                else:
                    dziecko = []
                    for x in range(0,self.dlugosc_indywidua,2) :
                        dziecko.append(meski[x])
                        if(self.dlugosc_indywidua > x+1):
                            dziecko.append(damski[x+1])
                    dzieci.append(dziecko)
                    if self.verbose:
                        print('meski  ',meski)
                        print('damski ', damski)
                        print(dziecko)

        mating_pool.extend(dzieci)
        return mating_pool


    def adopt_elite(self, population):
        local_population = list(population)
        for i in range(self.elite_num):
            population_loss = [self.fitness(x, self.target) for x in
                               local_population]

            best_index = population_loss.index(min(population_loss))
            if (self.fitness(self.elite[i], self.target) >
                population_loss[best_index] and local_population[best_index] not in self.elite):

                self.elite[i] = copy.deepcopy(local_population[best_index])
                del local_population[best_index]


    def evolve(self, populacja, cel):
        '''
        Funkcja ewoluujaca populacje aby uzyskac najlepszy wynik.
        :param populacja: ewoluowana populacja
        :param cel: zalozona wartosc celu
        :param zostaw: procent osobnikow, ktore przetrwaja
        :param zostaw_losowo: procent losowo wybieranych osobnikow
        :param mutuj: procent mutowanych osobnikow
        :return:
        '''

        mating_pool = self.selection(populacja)
        new_population = self.crossover(mating_pool)
        new_population = self.mutation(new_population)

        if self.elite_num != 0:
            self.adopt_elite(new_population)
            new_population.extend(self.elite)

        return new_population


    def run(self, iteracje, verbose=False):
        '''
        Rozwiązuje zadanie dla podanych parametrow
        :param populacja: zsprawdzana populacja
        :param cel: zalozona wartosc celu
        :return: zwracana jest lista z historia wartosci f-cji dopasowania
        '''

        file = open("wyniki.log", mode='a+')
        file.write("Proba %s.\n" %
                   (datetime.datetime.now()))


        for i in range(iteracje):
            self.population = self.evolve(self.population, self.target)

            # sprawdz czy sie polepsza
            average_loss = self.jakosc(self.population, self.target)
            self.fitness_history.append(average_loss)

            # sprawdz czy wygenerowano lepsze rozwiazanie, jak tak to zapamietaj
            best_p = self.best_from_population(self.population)
            if self.best is None: self.best = copy.deepcopy(best_p)

            if best_p[1] < self.best[1]:
                # zapamietaj najlepsze rozwiazanie
                self.best = copy.deepcopy(best_p)

            self.best_history.append(self.best[1])



            if verbose:
                print("iteracja: {iteracja}   average loss:{loss}\n"
                      .format(iteracja=i, loss=average_loss))
        file.close()

        return self.best


    def best_from_population(self, population):
        population_loss = [self.fitness(x, self.target) for x in population]
        best_index = population_loss.index(min(population_loss))
        return (population[best_index], population_loss[best_index])


    def worse_from_population(self, population):
        population_loss = [self.fitness(x, self.target) for x in population]
        best_index = population_loss.index(max(population_loss))
        return (population[best_index], population_loss[best_index])


if __name__ == '__main__':

    alg = AlgorytmGenetyczny(target=700, wielkosc_populacji=100, dlugosc_indywidua=10)
    history = alg.run(100)
    print(history)
