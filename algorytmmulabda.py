import numpy as np
import matplotlib.pyplot as plt


# Funkcja przystosowania
def funkcja_przystosowania(x1, x2):
    return np.sin(x1 * 0.05) + np.sin(x2 * 0.05) + 0.4 * np.sin(x1 * 0.15) * np.sin(x2 * 0.15)


# Inicjalizacja puli rodzicielskiej
def stworz_pule_rodzicielska(mu, xmin, xmax):
    return np.random.uniform(xmin, xmax, size=(mu, 2))


# Turniej - wybór najlepszego osobnika z grupy
def turniej(pula_rodzicielska, turniej_rozmiar):
    turniej_osobnicy = np.random.choice(len(pula_rodzicielska), size=turniej_rozmiar, replace=False)
    najlepszy_osobnik = max(turniej_osobnicy,
                             key=lambda i: funkcja_przystosowania(pula_rodzicielska[i, 0], pula_rodzicielska[i, 1]))
    return pula_rodzicielska[najlepszy_osobnik]


# Mutacja - dodanie losowej wartości
def mutacja(osobnik, mutacja_poziom, xmin, xmax):
    x1, x2 = osobnik
    x1 += np.random.uniform(-mutacja_poziom, mutacja_poziom)
    x2 += np.random.uniform(-mutacja_poziom, mutacja_poziom)
    x1 = np.clip(x1, xmin, xmax)
    x2 = np.clip(x2, xmin, xmax)
    return np.array([x1, x2])


# Selekcja - wybór najlepszych osobników
def selekcja(pula_rodzicielska, pula_potomna, mu):
    combined_pool = np.vstack((pula_rodzicielska, pula_potomna))
    fitness_values = np.array([funkcja_przystosowania(ind[0], ind[1]) for ind in combined_pool])
    selected_indices = np.argsort(fitness_values)[-mu:]
    return combined_pool[selected_indices]


# Algorytm ewolucyjny µ+λ
def algorytm_ewolucyjny(mu, lambda_, turniej_rozmiar, mutacja_poziom, iteracje_liczba, xmin, xmax):
    # Tworzenie początkowej puli rodzicielskiej
    pula_rodzicielska = stworz_pule_rodzicielska(mu, xmin, xmax)

    historia_punktow = []

    historia_punktow.append(pula_rodzicielska)
    for iteracja in range(iteracje_liczba):
        pula_potomna = []

        # Tworzenie puli potomnej
        for _ in range(lambda_):
            # Wybór osobnika za pomocą turnieju
            os_n = turniej(pula_rodzicielska, turniej_rozmiar)
            # Mutacja
            os_n_mutated = mutacja(os_n, mutacja_poziom, xmin, xmax)
            pula_potomna.append(os_n_mutated)

        # Selekcja najlepszych osobników
        pula_rodzicielska = selekcja(pula_rodzicielska, np.array(pula_potomna), mu)

        # Zapisywanie historii punktów dla wizualizacji
        historia_punktow.append(pula_rodzicielska)
    # Zwracamy historię punktów do rysowania wykresów
    return np.array(historia_punktow)


def rysuj_wykres(historia_punktow, xmin, xmax, iteracje_liczba, tytul):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    # Tworzenie siatki do rysowania funkcji przystosowania
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(xmin, xmax, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = funkcja_przystosowania(X1, X2)

    for i, ax in enumerate(axes):
        # Rysowanie izolini funkcji przystosowania
        CS = ax.contour(X1, X2, Z, 20, cmap='viridis')
        ax.clabel(CS, inline=True, fontsize=8)
        ax.set_title(f'Iteracja {i * (iteracje_liczba // 4)}')

        # Rysowanie punktów początkowych i dzieci w każdej iteracji
        if i == 0:  # 1. iteracja
            points = historia_punktow[0]
            ax.set_title(f'Przed iteracją')

            # Rysowanie rodziców jako kółka
            ax.scatter(points[:, 0], points[:, 1], color='r', marker='o', label='Rodzice')
        elif i == 1:  # 3. iteracja
            points = historia_punktow[3]
            ax.set_title(f'Iteracja: 3')

            # Rysowanie rodziców jako kółka
            ax.scatter(points[:, 0], points[:, 1], color='r', marker='o', label='Rodzice')
        elif i == 2:  # 13. iteracja
            points = historia_punktow[13]
            ax.set_title(f'Iteracja: 13')

            # Rysowanie rodziców jako kółka
            ax.scatter(points[:, 0], points[:, 1], color='r', marker='o', label='Rodzice')
        elif i == 3:  # 13. iteracja
            points = historia_punktow[iteracje_liczba - 1]
            ax.set_title(f'Iteracja: {iteracje_liczba - 1}')

            # Rysowanie rodziców jako kółka
            ax.scatter(points[:, 0], points[:, 1], color='r', marker='o', label='Rodzice')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.legend()

    plt.suptitle(tytul)
    plt.tight_layout()
    plt.show()


# Parametry dla pierwszego scenariusza
mu_1 = 4
lambda_1 = 3
turniej_rozmiar_1 = 2
mutacja_poziom_1 = 3
iteracje_liczba_1 = 20
xmin = 0
xmax = 100

# Uruchomienie algorytmu dla pierwszego scenariusza
historia_punktow_1 = algorytm_ewolucyjny(mu_1, lambda_1, turniej_rozmiar_1, mutacja_poziom_1, iteracje_liczba_1, xmin,
                                         xmax)

# Wizualizacja wyników dla pierwszego scenariusza
rysuj_wykres(historia_punktow_1, xmin, xmax, iteracje_liczba_1,
             f"Scenariusz 1: turniej_rozmiar=2, µ=4, λ=3, mutacja_poziom={mutacja_poziom_1}")

# Parametry dla drugiego scenariusza
mu_2 = 4
lambda_2 = 1
turniej_rozmiar_2 = 1
mutacja_poziom_2 = 3
iteracje_liczba_2 = 20

# Uruchomienie algorytmu dla drugiego scenariusza
historia_punktow_2 = algorytm_ewolucyjny(mu_2, lambda_2, turniej_rozmiar_2, mutacja_poziom_2, iteracje_liczba_2, xmin,
                                         xmax)

# Wizualizacja wyników dla drugiego scenariusza
rysuj_wykres(historia_punktow_2, xmin, xmax, iteracje_liczba_2,
             "Scenariusz 2: turniej_rozmiar=1, µ=4, λ=1, mutacja_poziom=3")
