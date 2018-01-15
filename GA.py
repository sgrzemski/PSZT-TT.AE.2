import random
from operator import add
from functools import reduce
import math
import numpy as np
import datetime
import copy

#self.coins = [1, 2, 5, 10, 20, 50, 100, 200, 500] #Self.Coins PLN, tylko calkowite

class GeneticAlgorythm:
    def __init__(self, target, coins, population_size=100, seed=None, leave=0.2,
                 random_leave=0.05, mutation=0.01, crossing_method=1, penalty_param1=1000,
                 penalty_param2=1.5, elite_num=2, verbose=False):

        random.seed(a=seed)

        self.target = target
        self.coins = coins
        self.population_size = population_size
        self.ind_length = len(self.coins)
        self.leave = leave
        self.random_leave = random_leave
        self.mutation = mutation
        self.crossing_method = crossing_method
        self.penalty_param1 = penalty_param1
        self.penalty_param2 = penalty_param2

        # init elite
        self.elite_num = elite_num
        self.elite = [self.create_individual() for e in range(self.elite_num)]

        # init population
        self.population = self.create_population(self.population_size)
        self.population.extend(self.elite)

        self.fitness_history = []  # log of mean population fitness
        self.best_history = []  # log of lowest loss value


        self.worse = None
        self.best = None
        # for test sake
        self.verbose = verbose


    def random_nominal_quantity(self, position):
        a = math.floor(self.target/self.coins[position])
        if a != 0:
            a = random.randint(0, a)
        return a


    def create_individual(self):
        '''
        Funkcja do generowania indywiduow.
        :param dlugosc: okresla ilosc elementow w indywiduale
        :return: zwraca iteral zawierajacy self.coins PLN
        '''
        list = []
        for x in range(self.ind_length) :
            list.append(self.random_nominal_quantity(x))
        return list


    def create_population(self, quantity):
        '''
        Funkcja do tworzenia populacji z indywiduow.
        :param ilesztuk: ilosc generowanych indywiduow w populacji
        :return: zwraca populacje zlozana z iteralow
        '''
        return [self.create_individual() for x in range(quantity)]


    def fitness(self, indywidual, cel):
        '''
        Funkcja sprawdzajaca bliskosc iteralu od zalozonego celu/reszty.
        :param indywidual: sprawdzane indywiduum
        :param cel: zalozona wartosc celu
        :return: zwraca wartosc bezwgledna sumy iteralu i celu
        '''
        # TODO: cross validate


        sum =  reduce(add, np.multiply(indywidual,self.coins))

        loss =abs(cel - sum)

        coin_quantity = reduce(add, indywidual)

        #print(self.coins)
        #print(indywidual)
        #print("Suma " ,suma)
        #print("Liczba monet" ,liczba_monet)

        # kara za nie osiagniecie targetu
        kara = loss * self.penalty_param1

        return self.penalty_param1 * loss**2 + self.penalty_param2*coin_quantity


    def quality(self, population, target):
        '''
        Funkcja sprawdzajaca jakosc calej populacji za pomoca funkcji fitness.
        :param populacja: zsprawdzana populacja
        :param cel: zalozona wartosc celu
        :return: zwracana jest lista jakosci dla kazdego indywidualu
        '''
        summed = reduce(add, (self.fitness(x, target) for x in population), 0)
        return summed / (len(population) * 1.0)


    def annihilate_popultaion(self):
        self.population = self.create_population(self.population_size)
        self.best_history = []
        self.fitness_history = []
        self.best = None
        self.worse = None
        self.elite = [self.create_individual() for e in range(self.elite_num)]



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
        leave_lenght = int(len(graded) * self.leave)

        #wyznacza z listy jakosci najlepsze iteraly
        parents = graded[:leave_lenght]

        #losowo wybierz inne indywidyaly zeby zapewnic rownosc genetyczna
        #z listy odrzuconych wybierz iteral
        for individual in graded[leave_lenght:]:
            if self.random_leave > random.random():
                parents.append(individual)

        return parents


    def do_mutation(self, population):
        '''Mutacja losowych osobników z populacji'''
        for individual in population:
            if self.mutation > random.random():
                pos_to_mutate = random.randint(0, len(individual) - 1)
                individual[pos_to_mutate] = self.random_nominal_quantity(pos_to_mutate)

        return population


    def crossover(self, mating_pool):
        ''' Krzyżowanie osobników wybranych do reprodukcji
        '''

        # crossover parents to create children
        parents_len = len(mating_pool)
        desired_len = self.population_size  - parents_len

        children = []
        while len(children) < desired_len:
            male = random.randint(0, parents_len - 1)
            female = random.randint(0, parents_len - 1)
            if male != female:
                male = mating_pool[male]
                female = mating_pool[female]
                if(self.crossing_method == 1):
                    half = int(len(male) / 2)
                    child = male[:half] + female[half:]
                    children.append(child)
                else:
                    child = []
                    for x in range(0,self.ind_length,2) :
                        child.append(male[x])
                        if(self.ind_length > x+1):
                            child.append(female[x+1])
                    children.append(child)
                    if self.verbose:
                        print('Male  ',male)
                        print('Female ', female)
                        print(child)

        mating_pool.extend(children)
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
        """
        Funkcja ewoluujaca populacje aby uzyskac najlepszy wynik.
        :param populacja: ewoluowana populacja
        :param cel: zalozona wartosc celu
        :param zostaw: procent osobnikow, ktore przetrwaja
        :param zostaw_losowo: procent losowo wybieranych osobnikow
        :param mutuj: procent mutowanych osobnikow
        :return:
        """

        mating_pool = self.selection(populacja)
        new_population = self.crossover(mating_pool)
        new_population = self.do_mutation(new_population)

        if self.elite_num != 0:
            self.adopt_elite(new_population)
            new_population.extend(self.elite)

        return new_population


    def run(self, iterations, verbose=False):
        """
        Rozwiązuje zadanie dla podanych parametrow
        :param populacja: zsprawdzana populacja
        :param cel: zalozona wartosc celu
        :return: zwracana jest lista z historia wartosci f-cji dopasowania
        """
        file = open("wyniki.log", mode='a+')
        file.write("Proba %s.\n" %
                   (datetime.datetime.now()))

        for i in range(iterations):
            self.population = self.evolve(self.population, self.target)

            # sprawdz czy sie polepsza
            average_loss = self.quality(self.population, self.target)
            self.fitness_history.append(average_loss)

            # sprawdz czy wygenerowano lepsze rozwiazanie, jak tak to zapamietaj
            best_p = self.best_from_population(self.population)
            if self.best is None: self.best = copy.deepcopy(best_p)

            if best_p[1] < self.best[1]:
                # zapamietaj najlepsze rozwiazanie
                self.best = copy.deepcopy(best_p)

            self.best_history.append(self.best[1])

            if verbose:
                print("iteracja: {iteration}   average loss:{loss}\n"
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

    alg = GeneticAlgorythm(target=700, population_size=100, ind_length=10)
    history = alg.run(100)
    print(history)
