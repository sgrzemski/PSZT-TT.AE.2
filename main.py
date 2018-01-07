import random
from operator import add
from functools import reduce

nominaly = [1, 2, 5, 10, 20, 50, 100, 200, 500] #Nominaly PLN, tylko calkowite


def stworz_indywidual(dlugosc):
    '''
    Funkcja do generowania indywiduow.
    :param dlugosc: okresla ilosc elementow w indywiduale
    :return: zwraca iteral zawierajacy nominaly PLN
    '''
    return [random.choice(nominaly) for x in range(dlugosc)]


def stworz_populacje(ilesztuk, dlugosc):
    '''
    Funkcja do tworzenia populacji z indywiduow.
    :param ilesztuk: ilosc generowanych indywiduow w populacji
    :param dlugosc: dlugosc pojedynczego indywiduum
    :return: zwraca populacje zlozana z iteralow
    '''
    return [stworz_indywidual(dlugosc) for x in range(ilesztuk)]


def fitness(indywidual, cel):
    '''
    Funkcja sprawdzajaca bliskosc iteralu od zalozonego celu/reszty.
    :param indywidual: sprawdzane indywiduum
    :param cel: zalozona wartosc celu
    :return: zwraca wartosc bezwgledna sumy iteralu i celu
    '''
    suma = reduce(add, indywidual, 0)
    return abs(cel - suma)


def jakosc(populacja, cel):
    '''
    Funkcja sprawdzajaca jakosc calej populacji za pomoca funkcji fitness.
    :param populacja: zsprawdzana populacja
    :param cel: zalozona wartosc celu
    :return: zwracana jest lista jakosci dla kazdego indywidualu
    '''
    summed = reduce(add, (fitness(x, cel) for x in populacja), 0)
    return summed / (len(populacja) * 1.0)


def ewoluuj(populacja, cel, zostaw=0.2, zostaw_losowo=0.05, mutuj=0.01):
    '''
    Funkcja ewoluujaca populacje aby uzyskac najlepszy wynik.
    :param populacja: ewoluowana populacja
    :param cel: zalozona wartosc celu
    :param zostaw: procent osobnikow, ktore przetrwaja
    :param zostaw_losowo: procent losowo wybieranych osobnikow
    :param mutuj: procent mutowanych osobnikow
    :return:
    '''
    graded = [(fitness(x, cel), x) for x in populacja] #tworzy liste jakosci dla kazdego indywiduum
    #na przyklad (180, [80, 75, 8, 85, 57])
    graded = [x[1] for x in sorted(graded)] #posortowana lista jakosci z usunietym fitnesem
    zostaw_dlugosc = int(len(graded) * zostaw) #wyznacza ilosc indywiduow, ktore przetrwaja (dlugosc listy graded razy procent pozostalych)
    rodzice = graded[:zostaw_dlugosc] #wyznacza z listy jakosci najlepsze iteraly

    #losowo wybierz inne indywidyaly zeby zapewnic rownosc genetyczna
    for individual in graded[zostaw_dlugosc:]:  #z listy odrzuconych wybierz iteral
        if zostaw_losowo > random.random():
            rodzice.append(individual)  #dodaj wybrany iteral do

    #mutacja losowych indywiduów z wybranej listy rodzicow
    for individual in rodzice:
        if mutuj > random.random():
            pos_to_mutate = random.randint(0, len(individual) - 1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = random.randint(
                min(individual), max(individual))

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

# ===============================
# TEST ALGORYTMU
# ===============================
target = 700
p_count = 100
i_length = 10
p = stworz_populacje(p_count, i_length)  # tworzy populacje 100 iteralow, kazdy po 5 liczb od 0 do 100
fitness_history = [jakosc(p, target)]  # zapisz jakosc populacji w liscie historii

for i in range(100):
    p = ewoluuj(p, target)
    fitness_history.append(jakosc(p, target))

for datum in fitness_history:
    print(datum)

# reszta = int(input("Wprowadz reszte do wydania:\n"))
# print("Wprowadzona reszta to: {}".format(reszta))
