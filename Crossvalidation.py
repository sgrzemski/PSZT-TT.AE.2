
# coding: utf-8

# In[23]:

# nessesary import
from GA import AlgorytmGenetyczny
import GA
import solver
import numpy as np
import matplotlib.pyplot as plt

import time


#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')


# In[25]:


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
        times.append(end-start)

    mean_time = np.mean(times)
    accuracy = np.sum(corrects) / len(corrects)

    if verbose:
        print("\t accuracy: {0}    mean time: {1:.8f}s"
              .format(accuracy, mean_time))

    return accuracy, mean_time


# In[26]:


params = dict()
seed = 'PSZT'
verbose = True

# Metody corssowania:
# 1: pół na pół; 2: naprzemiennie
params = {
    'wielkosc_populacji': 50,
    'elite_num': 2,
    'zostaw': 0.20,  # próg selekcji progowej
    'zostaw_losowo': 0.1,  # p-nstwo udzialu punktu w dalszej reprodukcji
    'mutuj': 0.5,
    'metoda_crossowania': 2
}

loss_params = {
    'kara_param': 100,
    'kara_param2': 1.5
}

vanilla=False
iteracje = 100
target = 200



# In[11]:

# sanity check - nasz algorytm powinien działać dla powyżysz/poniższych ustawień

coins=[1,2,5,10,20,50,100,200,500]

target = 420
seed = 'PSZT'

params['elite_num'] = 2
alg = AlgorytmGenetyczny(target, coins, **params, **loss_params, seed=seed,
                         verbose=False)

correct_solution = solver.coin_change_greedy(target, coins)
our_solution = alg.run(300)


print("correct: ", correct_solution)
print()
print("our: ", our_solution[0])
print("sum: ", np.sum(np.multiply(our_solution[0], coins)))
print("num: ", np.sum(our_solution[0]))



# In[ ]:


#test_algorithm(target,
#               algorythm=lambda: solver.coin_change_greedy(target, coins),
#               iterations=10,
#               verbose=True)


test_algorithm(target,
               algorythm=lambda: alg.run(iteracje),
               iterations=10,
               verbose=True)



# # Przykład 4fun, żeby pokazać naszą f-cje celu. 
# Rozważany jest przypadek 2d tzn mamy tylko 2 typy monet. Tak z szybkich testów
# to wynika, że dla aktualnych parametrów populacja mało różnorona i sporo
# osobników jest takich samych.
# 
# 

# In[13]:


target = 46
seed = 'PSZT'

coins = [3,5]

params['elite_num'] = 1
params['wielkosc_populacji'] = 100
alg = AlgorytmGenetyczny(target, coins, **params, **loss_params, seed=seed,
                         verbose=False)

alg.run(300)
pop_1 = alg.population

alg.run(300)
pop_2 = alg.population

our_solution = alg.run(300)
pop_3 = alg.population


print("our: ", our_solution[0])
print("sum: ", np.sum(np.multiply(our_solution[0], coins)))
print("num: ", np.sum(our_solution[0]))
print("loss: ", our_solution[1])


# In[20]:

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

fn = alg.fitness
max_x = math.floor(target/coins[1])+1  # liczba 10
max_y = math.floor(target/coins[0])+1  # liczba 5

x = np.arange(0, max_x, 1)
y = np.arange(0, max_y, 1)

X, Y = np.meshgrid(x, y)
z = np.zeros((max_y, max_x))
for i in range(max_x):
    for j in range(max_y):
        z[j,i] = fn([j,i], target)
        #print("z[{},{}]={}".format(j,i,fn([i,j],target)))


levels = [10, 100, 500, 2000, 4000, 20000, 100000]  # Levels must be increasing.
plt.figure(figsize=(10,7))
CS = plt.contour(x, y, z, 6,
                 colors='k',  # negative contours will be dashed by default
                 levels=levels
                 )
plt.clabel(CS, inline=1, fontsize=7, fmt='%0.4g')

points = list(zip(*pop_1))
points = [points[1], points[0]]
plt.scatter(*points, c='g')

points = list(zip(*pop_2))
points = [points[1], points[0]]
plt.scatter(*points, c='b')

points = list(zip(*pop_3))
points = [points[1], points[0]]
plt.scatter(*points, c='r')

plt.title('Przykład ewolucji populacji w przypadku dwuwymiarowym')
plt.legend(['300 iteracji', '600 iteracji', '900 iteracji'])
plt.show()


# # Eksperymentalne wyznaczenie parametrów.
# Najpierw mniej wiecej sie wyznaczy parametry, na podstawie uruchomień z
# małą liczbą iteracji.
# 

# In[40]:

target=437
coins=[1,2,5,10,20,50,100,200,500]


alg = AlgorytmGenetyczny(target, coins, **params, **loss_params, seed=seed,
                         verbose=False)

iterations = 500
parameters = [50, 100, 200, 500]


print("==========settings=============")
print("target: {} \ncoins: {}".format(target, coins))
print("iterations: {}".format(iterations))
print()
for param in parameters:
    alg.kara_param = param
    result = alg.run(iterations)

    print("+-+-+-+-+-parameters+-+-+-+-+-+")
    print("kara_param: ", alg.kara_param)
    print("----------results-------------")
    print("sum: ", np.sum(np.multiply(result[0], coins)))
    print("coin num: ", np.sum(result[0]))
    print("loss: ", result[1])
    print()


