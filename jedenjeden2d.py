import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Funkcja przystosowania
def funkcja_przystosowania(x1, x2):
    return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)


# Inicjalizacja rozwiązania: Losowa pozycja
def inicjalizuj_rozwiazanie(xmin, xmax):
    return np.random.uniform(xmin, xmax, 2)  # Losowe początkowe rozwiązanie


# Algorytm 1+1
def algorytm_1_plus_1(iteracje_liczba, xmin, xmax):
    # Inicjalizacja
    X = inicjalizuj_rozwiazanie(xmin, xmax)  # Początkowe rozwiązanie
    y = funkcja_przystosowania(X[0], X[1])  # Wartość funkcji przystosowania dla początkowego rozwiązania

    # Historia rozwiązania do wykresu
    X_history = [np.copy(X)]

    for iteracja in range(iteracje_liczba):
        # Tworzenie perturbacji
        perturbacja = np.random.uniform(-1, 1, 2)  # Losowa zmiana
        X_new = X + perturbacja  # Nowe rozwiązanie

        # Ograniczenie w obrębie dopuszczalnego zakresu
        X_new = np.clip(X_new, xmin, xmax)

        # Ocena nowego rozwiązania
        y_new = funkcja_przystosowania(X_new[0], X_new[1])

        # Jeśli nowe rozwiązanie jest lepsze, przyjmujemy je
        if y_new > y:
            X = X_new
            y = y_new

        # Zapisanie rozwiązania do historii
        X_history.append(np.copy(X))

    return X_history


def rysuj_wykres_konturowy_2d(X_history, xmin, xmax):
    # Tworzenie siatki punktów
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Obliczanie wartości funkcji przystosowania dla siatki
    Z = funkcja_przystosowania(X1, X2)

    # Tworzenie wykresu 2D
    fig, ax = plt.subplots(figsize=(8, 6))

    # Rysowanie konturów funkcji
    contour = ax.contourf(X1, X2, Z, 50, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Wartości funkcji')

    # Rysowanie trajektorii punktów optymalizacyjnych
    X_history = np.array(X_history)
    ax.plot(X_history[:, 0], X_history[:, 1], color='r', marker='o', markersize=5, label="Trajektoria")

    # Dodanie etykiet do punktów na trajektorii (co 10. punkt)
    for i in range(0, len(X_history), 20):
        y_val = funkcja_przystosowania(X_history[i, 0], X_history[i, 1])
        ax.text(X_history[i, 0], X_history[i, 1], f'{i}', color='black', fontsize=8, ha='center')

    # Etykiety osi i tytuł
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title("Algorytm 1+1 - Optymalizacja funkcji (Wykres 2D)")

    # Wyświetlenie legendy
    ax.legend()
    plt.show()


# Wykres 3D
def rysuj_wykres_3d(X_history, xmin, xmax):
    # Tworzenie siatki punktów
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Obliczanie wartości funkcji przystosowania dla siatki
    Z = funkcja_przystosowania(X1, X2)

    # Tworzenie wykresu 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Rysowanie powierzchni
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='k', alpha=0.6)

    # Rysowanie trajektorii punktów optymalizacyjnych w przestrzeni 3D
    X_history = np.array(X_history)
    ax.plot(X_history[:, 0], X_history[:, 1],
            [funkcja_przystosowania(X_history[i, 0], X_history[i, 1]) for i in range(len(X_history))], color='r',
            marker='o', markersize=5, label="Trajektoria")

    # Etykiety osi i tytuł
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Funkcja przystosowania')
    ax.set_title("Algorytm 1+1 - Optymalizacja funkcji (Wykres 3D)")
    ax.legend()
    plt.show()


# Parametry algorytmu
xmin = 0
xmax = 100
iteracje_liczba = 300

# Uruchomienie algorytmu 1+1
X_history = algorytm_1_plus_1(iteracje_liczba, xmin, xmax)

# Wykres konturowy 2D z siatką
rysuj_wykres_konturowy_2d(X_history, xmin, xmax)

# Wykres 3D
rysuj_wykres_3d(X_history, xmin, xmax)
