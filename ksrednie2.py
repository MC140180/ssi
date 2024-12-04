import numpy as np
import matplotlib.pyplot as plt


def odleglosc(pktA, pktB):
    return np.abs(pktA[0] - pktB[0])


def przypisz_przypki_do_grup(data, centra):
    przynaleznosc = np.zeros(len(data), dtype=int)
    for i, probka in enumerate(data):
        odleglosci = np.array([odleglosc(probka, centrum) for centrum in centra])
        przynaleznosc[i] = np.argmin(odleglosci)
    return przynaleznosc


def aktualizuj_srodki(data, przynaleznosc, k):
    nowe_centra = []
    for j in range(k):
        probki_w_grupie = data[przynaleznosc == j]
        if len(probki_w_grupie) > 0:
            nowe_centra.append(np.mean(probki_w_grupie, axis=0))
        else:
            nowe_centra.append(np.random.choice(data, size=1).flatten())  # Zapobieganie pustej grupie
    return np.array(nowe_centra)


def k_srednich(data, k, iters):

    # Losowy wybór początkowych centrów
    np.random.seed(42)
    poczatkowe_centra_idx = np.random.choice(len(data), k, replace=False)
    centra = data[poczatkowe_centra_idx]

    historia_centrow = []

    for iteracja in range(iters):
        # 1. Przypisz próbki do najbliższego środka
        przynaleznosc = przypisz_przypki_do_grup(data, centra)

        # Zapisz historię centrów
        historia_centrow.append(centra.copy())

        # 2. Aktualizuj środki grup
        centra = aktualizuj_srodki(data, przynaleznosc, k)

    return przynaleznosc, historia_centrow, centra


# Wczytanie danych danych
data = np.loadtxt("probki.txt")
k = 4
iters = 10


przynaleznosc, historia_centrow, finalne_centra = k_srednich(data, k, iters)


# Raportowanie wyników
def raport(data, przynaleznosc, centra, iteracja, k):
    print(f"--- Raport dla iteracji {iteracja} ---")
    for j in range(k):
        probki_w_grupie = data[przynaleznosc == j]
        if len(probki_w_grupie) > 0:
            print(f"Grupa {j + 1}:")
            print(f"  Środek: {centra[j]}")
            print(f"  Liczba próbek: {len(probki_w_grupie)}")
            print(f"  x1 min: {probki_w_grupie[:, 0].min()}, x1 max: {probki_w_grupie[:, 0].max()}")
            print(f"  x2 min: {probki_w_grupie[:, 1].min()}, x2 max: {probki_w_grupie[:, 1].max()}")


# Raport po 4 i 10 iteracjach
raport(data, przynaleznosc, historia_centrow[3], 4, k)
raport(data, przynaleznosc, historia_centrow[-1], 10, k)



def rysuj_wykres(data, przynaleznosc, centra, iteracja, k):
    plt.figure(figsize=(8, 8))
    for j in range(k):
        probki_w_grupie = data[przynaleznosc == j]
        plt.scatter(probki_w_grupie[:, 0], probki_w_grupie[:, 1], label=f'Grupa {j + 1}')
    plt.scatter(centra[:, 0], centra[:, 1], color='red', marker='x', s=200, label='Środki')
    plt.title(f"K-średnich: iteracja {iteracja}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Wykresy po 4 i 10 iteracjach
rysuj_wykres(data, przynaleznosc, historia_centrow[3], 4, k)
rysuj_wykres(data, przynaleznosc, historia_centrow[iters-1], iters, k)
