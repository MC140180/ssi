import numpy as np
import matplotlib.pyplot as plt


# Funkcja obliczająca miarę euklidesową
def odleglosc_euklidesowa(pktA, pktB):
    return np.sqrt(np.sum((pktA - pktB) ** 2))


# Algorytm Fuzzy c-Means
def fuzzy_c_means(data, k, iters, fcm_m=2, epsilon=1e-5):
    M, n = data.shape
    np.random.seed(42)

    # 1. Inicjalizacja
    U = np.random.rand(k, M)
    U /= np.sum(U, axis=0)  # Normalizacja stopni przynależności
    V = np.random.rand(k, n)  # Losowe środki grup
    historia_centrow = [V.copy()]  # Dodajemy początkowe centra do historii

    for iteracja in range(iters):
        # 2.1 Obliczanie odległości
        D = np.zeros((k, M))
        for j in range(k):
            for s in range(M):
                D[j, s] = odleglosc_euklidesowa(data[s], V[j])
        D[D < epsilon] = epsilon  # Unikamy zerowych wartości

        # 2.3 Wyliczenie stopnia przynależności U
        for j in range(k):
            for s in range(M):
                U[j, s] = 1 / np.sum((D[j, s] / D[:, s]) ** (2 / (fcm_m - 1)))

        if np.any(np.isnan(U)):
            raise ValueError("Macierz U zawiera wartości nieoznaczone. Przerwanie algorytmu.")

        # 2.5 Obliczenie nowych środków V
        nowe_V = np.zeros_like(V)
        for j in range(k):
            um_powered = U[j] ** fcm_m
            nowe_V[j] = np.sum(um_powered[:, np.newaxis] * data, axis=0) / np.sum(um_powered)
        historia_centrow.append(nowe_V.copy())
        V = nowe_V

    return U, V, historia_centrow


# Funkcja raportująca
def raport_fcm(data, U, centra, iteracja, k, fcm_m=2):
    print(f"--- Raport dla iteracji {iteracja} ---")
    for j in range(k):
        przynaleznosc_mask = U[j] > 0.6
        probki_w_grupie = data[przynaleznosc_mask]
        if len(probki_w_grupie) > 0:
            print(f"Grupa {j + 1}:")
            print(f"  Środek: {centra[j]}")
            print(f"  Liczba próbek z przynależnością > 0.6: {len(probki_w_grupie)}")
            print(f"  x1 min: {probki_w_grupie[:, 0].min()}, x1 max: {probki_w_grupie[:, 0].max()}")
            print(f"  x2 min: {probki_w_grupie[:, 1].min()}, x2 max: {probki_w_grupie[:, 1].max()}")


# Wizualizacja
def rysuj_wykres_fcm(data, U, centra, iteracja, k, tytul=""):
    plt.figure(figsize=(8, 8))
    for j in range(k):
        przynaleznosc_mask = U[j] > 0.6
        probki_w_grupie = data[przynaleznosc_mask]
        plt.scatter(probki_w_grupie[:, 0], probki_w_grupie[:, 1], label=f'Grupa {j + 1}')
    plt.scatter(centra[:, 0], centra[:, 1], color='red', marker='X', s=200, label='Środki')
    plt.title(f"{tytul} (Iteracja {iteracja})")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Wykres początkowy
def rysuj_wykres_poczatkowy(data, centra, k):
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], label='Próbki', alpha=0.5)
    plt.scatter(centra[:, 0], centra[:, 1], color='red', marker='X', s=200, label='Początkowe środki')
    plt.title("Fuzzy c-Means: Przed iteracjami")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


# Wczytanie danych
data = np.loadtxt("probki.txt")
k = 3
iters = 20
fcm_m = 2

# Losowanie początkowych środków
np.random.seed(42)
M, n = data.shape
V_init = np.random.rand(k, n)

# Wykres początkowy
rysuj_wykres_poczatkowy(data, V_init, k)

# Uruchomienie algorytmu
U, finalne_centra, historia_centrow = fuzzy_c_means(data, k, iters, fcm_m) # U macierz stopnia przynaleznosci probek


# Raport po 4 i 20 iteracjach
raport_fcm(data, U, historia_centrow[3], 4, k)
raport_fcm(data, U, historia_centrow[-1], 20, k)

# Wykresy po 4 i 20 iteracjach
rysuj_wykres_fcm(data, U, historia_centrow[3], 4, k, tytul=f"Stan po 4 iteracji")
rysuj_wykres_fcm(data, U, historia_centrow[iters-1], iters, k, tytul="Stan po 20 iteracjach")
print(U)