import random
from operator import add
from functools import reduce
import math
import numpy as np

nominaly = [1, 2, 5, 10, 20, 50, 100, 200, 500] #Nominaly PLN, tylko calkowite

class AlgorytmGenetyczny:
    def __init__(self, target, wielkosc_populacji, seed=None, vanilla=False):
        '''
        Inicjacja algorytmu
        :param target: kwota do wydania
        :param wielkosc_populacji: wielkość generowanej co iteracje populacji
        :param dlugosc_indywidua: maksymalna liczba wydawanych monet
        '''

        self.target = target
        self.wielkosc_populacji = wielkosc_populacji
        self.dlugosc_indywidua = len(nominaly)

        random.seed(a=seed)

        # najlepsze rozwiazanie
        self.best = [0, 10000]

        # for test sake
        self.vanilla = vanilla


    def stworz_indywidual(self):
        '''
        Funkcja do generowania indywiduow.
        :param dlugosc: okresla ilosc elementow w indywiduale
        :return: zwraca iteral zawierajacy nominaly PLN
        '''
        list = []
        for x in range(self.dlugosc_indywidua) :
            a = math.floor(self.target/nominaly[x])
            if a == 0:
                list.append(0)
            else:
                list.append(random.randint(1, a))

        #print(list)
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
        kara_param = 1000

        suma =  reduce(add, np.multiply(indywidual,nominaly))

        loss =abs(cel - suma)

        liczba_monet = reduce(add, indywidual)

        #print(nominaly)
        #print(indywidual)
        #print("Suma " ,suma)
        #print("Liczba monet" ,liczba_monet)

        # kara za nie osiagniecie targetu
        kara = loss * kara_param

        if self.vanilla:
            liczba_monet = 0
            kara_param = 1

        return kara_param * loss**2 + liczba_monet


    def jakosc(self, populacja, cel):
        '''
        Funkcja sprawdzajaca jakosc calej populacji za pomoca funkcji fitness.
        :param populacja: zsprawdzana populacja
        :param cel: zalozona wartosc celu
        :return: zwracana jest lista jakosci dla kazdego indywidualu
        '''
        summed = reduce(add, (self.fitness(x, cel) for x in populacja), 0)
        return summed / (len(populacja) * 1.0)


    def ewoluuj(self, populacja, cel, zostaw=0.2, zostaw_losowo=0.05, mutuj=0.01):
        '''
        Funkcja ewoluujaca populacje aby uzyskac najlepszy wynik.
        :param populacja: ewoluowana populacja
        :param cel: zalozona wartosc celu
        :param zostaw: procent osobnikow, ktore przetrwaja
        :param zostaw_losowo: procent losowo wybieranych osobnikow
        :param mutuj: procent mutowanych osobnikow
        :return:
        '''
        #tworzy liste jakosci dla kazdego indywiduum
        #na przyklad (180, [80, 75, 8, 85, 57])
        graded = [(self.fitness(x, cel), x) for x in populacja]

        #posortowana lista jakosci z usunietym fitnesem
        graded = [x[1] for x in sorted(graded)]

        # wd wyznacza ilosc indywiduow, ktore przetrwaja (dlugosc listy graded razy
        # procent pozostalych)
        zostaw_dlugosc = int(len(graded) * zostaw)

        #wyznacza z listy jakosci najlepsze iteraly
        rodzice = graded[:zostaw_dlugosc]

        #losowo wybierz inne indywidyaly zeby zapewnic rownosc genetyczna
        #z listy odrzuconych wybierz iteral
        for individual in graded[zostaw_dlugosc:]:
            if zostaw_losowo > random.random():
                rodzice.append(individual)  #dodaj wybrany iteral do

        #mutacja losowych indywiduów z wybranej listy rodzicow
        for individual in rodzice:
            if mutuj > random.random():
                pos_to_mutate = random.randint(0, len(individual) - 1)
                individual[pos_to_mutate] = random.choice(nominaly)

        # crossover parents to create children
        rodzice_dlugosc = len(rodzice)
        rzadana_dlugosc = len(populacja) - rodzice_dlugosc
        dzieci = []
        while len(dzieci) < rzadana_dlugosc:
            meski = random.randint(0, rodzice_dlugosc - 1)
            damski = random.randint(0, rodzice_dlugosc - 1)
            if meski != damski:
                meski = rodzice[meski]
                damski = rodzice[damski]
                polowa = int(len(meski) / 2)
                dziecko = meski[:polowa] + damski[polowa:]
                dzieci.append(dziecko)

        rodzice.extend(dzieci) #Dodaj dzieci do zbioru rodzicow
        return rodzice

    def run(self, iteracje, verbose=False):
        '''
        Rozwiązuje zadanie dla podanych parametrow
        :param populacja: zsprawdzana populacja
        :param cel: zalozona wartosc celu
        :return: zwracana jest lista z historia wartosci f-cji dopasowania
        '''


        # tworzy populacje 100 iteralow, kazdy po 5 liczb od 0 do 100
        p = self.stworz_populacje(self.wielkosc_populacji)

        # zapisz jakosc populacji w liscie historii
        self.fitness_history = [self.jakosc(p, self.target)]

        for i in range(iteracje):
            p = self.ewoluuj(p, self.target)

            # sprawdz czy wygenerowano lepsze rozwiazanie, jak tak to zapamietaj
            best_p = self.best_from_population(p)
            if best_p[1] < self.best[1]:
                # zapamietaj najlepsze rozwiazanie
                self.best = best_p

            # sprawdz czy sie polepsza
            average_loss = self.jakosc(p, self.target)
            self.fitness_history.append(average_loss)

            if verbose:
                print("iteracja: {iteracja}   average loss:{loss}"
                      .format(iteracja=i, loss=average_loss))


        return self.fitness_history


    def best_from_population(self, population):
        population_loss = [self.fitness(x, self.target) for x in population]
        best_index = population_loss.index(min(population_loss))
        return (population[best_index], population_loss[best_index])


if __name__ == '__main__':

    alg = AlgorytmGenetyczny(target=700, wielkosc_populacji=100, dlugosc_indywidua=10)
    history = alg.run(100)
    print(history)
