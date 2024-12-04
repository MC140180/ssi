import numpy as np
import matplotlib.pyplot as plt


# Funkcja przystosowania
def funkcja_przystosowania(x1, x2):
    return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)


# Inicjalizacja populacji
def inicjalizuj_populacje(N, xmin, xmax, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.uniform(xmin, xmax, (N, 2))  # Pozycje
    V = np.zeros_like(X)  # Prędkości
    Xlok = np.copy(X)  # Lokalne optima
    return X, V, Xlok


# Ocena populacji
def oceń_populację(X):
    return np.array([funkcja_przystosowania(x[0], x[1]) for x in X])


# Aktualizacja lokalnych i globalnych optima
def aktualizuj_lokalne_optimum(X, Xlok, y):
    for j in range(len(X)):
        if y[j] > funkcja_przystosowania(Xlok[j, 0], Xlok[j, 1]):
            Xlok[j] = X[j]
    return Xlok


def aktualizuj_globalne_optimum(X, y, xglob, yglob):
    najlepszy_indeks = np.argmax(y)
    if y[najlepszy_indeks] > yglob:
        xglob = X[najlepszy_indeks]
        yglob = y[najlepszy_indeks]
    return xglob, yglob


# Aktualizacja prędkości i pozycji
def aktualizuj_prędkość(V, X, Xlok, xglob, rglob, rlok, rinercji):
    rndglob = np.random.rand(2)
    rndlok = np.random.rand(2)
    for j in range(len(X)):
        V[j] = rinercji * V[j] + rglob * rndglob * (xglob - X[j]) + rlok * rndlok * (Xlok[j] - X[j])
    return V


def aktualizuj_pozycje(X, V, xmin, xmax):
    X += V
    return np.clip(X, xmin, xmax)


# Algorytm PSO
def pso(N, rglob, rlok, rinercji, xmin, xmax, iteracje_liczba, seed=None):
    X, V, Xlok = inicjalizuj_populacje(N, xmin, xmax, seed=seed)
    y = oceń_populację(X)
    xglob = X[np.argmax(y)]
    yglob = np.max(y)
    historia = [(np.copy(X), xglob, yglob)]  # Historia

    for iteracja in range(iteracje_liczba):
        Xlok = aktualizuj_lokalne_optimum(X, Xlok, y)
        xglob, yglob = aktualizuj_globalne_optimum(X, y, xglob, yglob)
        V = aktualizuj_prędkość(V, X, Xlok, xglob, rglob, rlok, rinercji)
        X = aktualizuj_pozycje(X, V, xmin, xmax)
        y = oceń_populację(X)
        historia.append((np.copy(X), xglob, yglob))

    return historia


# Rysowanie punktów
def rysuj_punkty(historia, xmin, xmax, iteracja, tytuł):
    X, xglob, yglob = historia[iteracja]
    y = oceń_populację(X)

    najlepszy = X[np.argmax(y)]
    najgorszy = X[np.argmin(y)]

    # Generowanie siatki
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = funkcja_przystosowania(X1, X2)

    plt.figure(figsize=(10, 6))
    plt.contourf(X1, X2, Z, levels=20, cmap="RdGy", alpha=0.8)
    plt.colorbar(label="Wartość funkcji przystosowania")

    # Rysowanie najlepszych i najgorszych punktów
    plt.scatter(najlepszy[0], najlepszy[1], color="green", s=150, marker="*", label="Najlepszy punkt")
    plt.scatter(najgorszy[0], najgorszy[1], color="red", s=150, marker="x", label="Najgorszy punkt")

    plt.title(tytuł)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()


# Scenariusz 1: rglob = 1, rinercji = 0, rlok = 0
xmin, xmax = 0, 100
N, iteracje = 4, 30
seed = 42

historia1 = pso(N, rglob=1.0, rlok=0.0, rinercji=0.0, xmin=xmin, xmax=xmax, iteracje_liczba=iteracje, seed=seed)
rysuj_punkty(historia1, xmin, xmax, iteracja=0, tytuł="Początkowe położenie (Scenariusz 1)")
rysuj_punkty(historia1, xmin, xmax, iteracja=5, tytuł="Po 5 iteracjach (Scenariusz 1)")

# Scenariusz 2: rglob = 1, rinercji = 0.7, rlok = 0
historia2 = pso(N, rglob=1.0, rlok=0.0, rinercji=0.7, xmin=xmin, xmax=xmax, iteracje_liczba=iteracje, seed=seed)
rysuj_punkty(historia2, xmin, xmax, iteracja=0, tytuł="Początkowe położenie (Scenariusz 2)")
rysuj_punkty(historia2, xmin, xmax, iteracja=5, tytuł="Po 5 iteracjach (Scenariusz 2)")

# Scenariusz 3: rglob = 0.05, rinercji = 0, rlok = 0.8, N = 20
N = 20
historia3 = pso(N, rglob=0.05, rlok=0.8, rinercji=0.0, xmin=xmin, xmax=xmax, iteracje_liczba=13)
rysuj_punkty(historia3, xmin, xmax, iteracja=0, tytuł="Początkowe położenie (Scenariusz 3)")
rysuj_punkty(historia3, xmin, xmax, iteracja=3, tytuł="Po 3 iteracjach (Scenariusz 3)")
rysuj_punkty(historia3, xmin, xmax, iteracja=13, tytuł="Po 13 iteracjach (Scenariusz 3)")
