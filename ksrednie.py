import numpy as np
import matplotlib.pyplot as plt


# Funkcje obliczające różne miary odległości
def odleglosc_euklidesowa(pktA, pktB):
    return np.sqrt(np.sum((pktA - pktB) ** 2))


def odleglosc_manhattan(pktA, pktB):
    return np.sum(np.abs(pktA - pktB))


def odleglosc_czebyszew(pktA, pktB):
    return np.max(np.abs(pktA - pktB))


def odleglosc_minkowskiego(pktA, pktB, mink):
    if mink <= 0:
        raise ValueError("Parametr 'mink' musi być większy od 0.")
    return np.sum(np.abs(pktA - pktB) ** mink) ** (1 / mink)


def wybierz_miare_odleglosci(nazwa_miary, mink=None):
    if nazwa_miary == "minkowski":
        return lambda pktA, pktB: odleglosc_minkowskiego(pktA, pktB, mink)
    miary = {
        "euklidesowa": odleglosc_euklidesowa,
        "manhattan": odleglosc_manhattan,
        "czebyszew": odleglosc_czebyszew,
    }
    return miary.get(nazwa_miary, odleglosc_euklidesowa)


def przypisz_przypki_do_grup(data, centra, funkcja_odleglosci):
    przynaleznosc = np.zeros(len(data), dtype=int)
    for i, probka in enumerate(data):
        odleglosci = np.array([funkcja_odleglosci(probka, centrum) for centrum in centra])
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


def k_srednich(data, k, iters, nazwa_miary="euklidesowa", mink=None):
    # Wybór funkcji odległości
    funkcja_odleglosci = wybierz_miare_odleglosci(nazwa_miary, mink)

    # Losowy wybór początkowych centrów
    np.random.seed(42)
    poczatkowe_centra_idx = np.random.choice(len(data), k, replace=False)
    centra = data[poczatkowe_centra_idx]

    historia_centrow = []

    for iteracja in range(iters):
        # 1. Przypisz próbki do najbliższego środka
        przynaleznosc = przypisz_przypki_do_grup(data, centra, funkcja_odleglosci)

        # Zapisz historię centrów
        historia_centrow.append(centra.copy())

        # 2. Aktualizuj środki grup
        centra = aktualizuj_srodki(data, przynaleznosc, k)

    return przynaleznosc, historia_centrow, centra


# Wczytanie danych
data = np.loadtxt("probki.txt")
k = 3
iters = 10
nazwa_miary = "euklidesowa"  # Możliwe: "euklidesowa", "manhattan", "czebyszew", "minkowski"
mink_param = 3  # Parametr Minkowskiego (dla miary Minkowskiego)


przynaleznosc, historia_centrow, finalne_centra = k_srednich(data, k, iters, nazwa_miary, mink_param)


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
    plt.scatter(centra[:, 0], centra[:, 1], color='red', marker='X', s=200, label='Środki')
    plt.title(f"K-średnich: iteracja {iteracja}")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Wykresy po 4 i 10 iteracjach
rysuj_wykres(data, przynaleznosc, historia_centrow[3], 4, k)
rysuj_wykres(data, przynaleznosc, historia_centrow[iters - 1], iters, k)
