from GA import AlgorytmGenetyczny


def main():
    seed = "PSZT"

    alg = AlgorytmGenetyczny(target=700, wielkosc_populacji=100,
                             dlugosc_indywidua=10, seed=seed)
    history = alg.run(100)
    print(history)


if __name__ == '__main__':
    main()
