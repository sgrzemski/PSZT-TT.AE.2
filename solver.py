import timeit


def coin_change_greedy(target, coins):
    ''' Coin change problem solution with use of greedy strategy
    https://pl.wikipedia.org/wiki/Problem_wydawania_reszty
    '''
    solution = []
    while target > 0:
        n = 0
        for i in range(len(coins)):
            if (coins[i] <= target) and (coins[i] > n):
                n = coins[i]
        target -= n
        solution.append(n)

    return solution


def coin_change_dp(target, coins):
    ''' Dynamic programming solution to coin change problem
    code taken from:
    http://interactivepython.org/runestone/static/pythonds/Recursion/DynamicProgramming.html

    TODO: refator bo wtf i ogarnąć ocb here
    https://pl.wikipedia.org/wiki/Problem_wydawania_reszty
    '''

    minCoins = [0]*(target+1)
    coinsUsed = [0]*(target+1)

    for cents in range(target+1):
        coinCount = cents
        newCoin = 1
        for j in [c for c in coins if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
                coinCount = minCoins[cents-j]+1
                newCoin = j
            minCoins[cents] = coinCount
            coinsUsed[cents] = newCoin

    solution =[]
    while target > 0:
        thisCoin = coinsUsed[target]
        solution.append(thisCoin)
        target = target - thisCoin

    return solution

def time_solutions(algorythms, target, coins, iterations=10):
    #coins = [1,2,5,10,20,50,100,200,500]
    solutions = []
    timings = []
    print(algorythms)

    for alg in algorythms:
        print(alg)
        solutions.append(alg(target, coins))
        exec('alg')
        timings.append(timeit.timeit('alg(target, coins)',
                                 number=iterations))

    return solutions, timmings


if __name__ == '__main__':

    target = 555
    coins = [1,2,5,10,20,50,100,200]
    solution_dp = coin_change_dp(target, coins)
    solution_greedy = coin_change_greedy(target, coins)


    greedy_time = timeit.timeit('coin_change_greedy(target, coins)',
                                globals=globals(), number=iterations)

    dp_time = timeit.timeit('coin_change_dp(target, coins)', globals=globals(),
                                number=iterations)

    print("greedy solution: coin_count={coins}, time={time:.8f}"
          .format(coins=len(solution_greedy), time=greedy_time))

    print("dp solution: coin_count={coins}, time={time:.8f}"
          .format(coins=len(solution_dp), time=dp_time))

    #print("Making change for {target} requires {length} coins."
    #      .format(target=target, length=len(solution)))
    #print("Solution is: {solution}".format(solution=solution))
