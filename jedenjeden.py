import numpy as np
import matplotlib.pyplot as plt


# Funkcja przystosowania
def funkcja_przystosowania(x):
    return np.sin(x / 10.0) * np.sin(x / 200.0)


# Funkcja do losowania wartości x z przedziału
def losuj_x(zakres_zmienności):
    return np.random.uniform(zakres_zmienności[0], zakres_zmienności[1])


# Funkcja do korekcji x, aby nie wyszło poza zakres
def skoryguj_x(xpot, zakres_zmienności):
    if xpot < zakres_zmienności[0]:
        return zakres_zmienności[0]
    elif xpot > zakres_zmienności[1]:
        return zakres_zmienności[1]
    return xpot


# Funkcja do aktualizacji rozrzutu w zależności od wyników
def aktualizuj_rozrzut(rozrzut, wsp_przyrostu, lepszy_wynik):
    if lepszy_wynik:
        return rozrzut * wsp_przyrostu  # Zwiększ rozrzut
    else:
        return rozrzut / wsp_przyrostu  # Zmniejsz rozrzut


# Funkcja do przeprowadzenia jednej iteracji algorytmu 1+1
def algorytm_1_1_iteracja(x, y, rozrzut, wsp_przyrostu, zakres_zmienności):
    # Losowanie nowego xpot w obrębie rozrzutu
    xpot = x + np.random.uniform(-rozrzut, rozrzut)

    # Korekcja xpot, jeśli wychodzi poza zakres
    xpot = skoryguj_x(xpot, zakres_zmienności)

    # Obliczenie nowego ypot
    ypot = funkcja_przystosowania(xpot)

    # Porównanie wyników i decyzja o aktualizacji
    lepszy_wynik = ypot >= y
    if lepszy_wynik:
        x, y = xpot, ypot

    # Aktualizacja rozrzutu
    rozrzut = aktualizuj_rozrzut(rozrzut, wsp_przyrostu, lepszy_wynik)

    return x, y, rozrzut, lepszy_wynik


# Główna funkcja algorytmu 1+1
def algorytm_1_1(rozrzut, wsp_przyrostu, l_iteracji, zakres_zmienności):
    # Inicjalizacja
    x = losuj_x(zakres_zmienności)
    y = funkcja_przystosowania(x)

    # Wyniki
    x_values = [x]
    y_values = [y]
    rozrzut_values = [rozrzut]

    # Pętla algorytmu
    for iteracja in range(l_iteracji):
        x, y, rozrzut, _ = algorytm_1_1_iteracja(x, y, rozrzut, wsp_przyrostu, zakres_zmienności)

        # Zapisywanie wyników
        x_values.append(x)
        y_values.append(y)
        rozrzut_values.append(rozrzut)

    return x_values, y_values, rozrzut_values


# Funkcja rysująca wykres
def rysuj_wykres(x_values, y_values, rozrzut_values, zakres_zmienności):
    x = np.linspace(zakres_zmienności[0], zakres_zmienności[1], 1000)
    y = funkcja_przystosowania(x)

    plt.figure(figsize=(10, 6))

    # Wykres funkcji przystosowania
    plt.plot(x, y, label="Funkcja przystosowania", color="blue")

    # Wykres punktów
    plt.scatter(x_values, y_values, marker='o', linestyle='-', color='red', label="Znalezione punkty")

    # Dodanie etykiet iteracji
    for i, (xi, yi) in enumerate(zip(x_values, y_values)):
        if i % 5 == 0:  # Pomijamy pierwszy punkt, bo to punkt początkowy
            plt.text(xi, yi, str(i), fontsize=9, color="black", ha="left", va="bottom")

    # Tytuł i etykiety
    plt.title("Algorytm 1+1 - Wyniki optymalizacji")
    plt.xlabel("x")
    plt.ylabel("y (funkcja przystosowania)")
    plt.legend()

    plt.grid(True)
    plt.show()


# Przykłady z różnych warunków początkowych

# Parametry
rozrzut = 10
wsp_przyrostu = 1.1
l_iteracji = 20
zakres_zmienności = [0, 100]

# 1. Wylosowanie początkowego x z zakresu dostępności
x_values_1, y_values_1, rozrzut_values_1 = algorytm_1_1(rozrzut, wsp_przyrostu, l_iteracji, zakres_zmienności)
print("Przykład 1 - x, y, rozrzut na początku, po 5, 10, 15 iteracjach:")
print(f"Na początku: x = {x_values_1[0]:.4f}, y = {y_values_1[0]:.4f}")
for i in [5, 10, 15]:
    print(f"Po {i} iteracjach: x = {x_values_1[i]:.4f}, y = {y_values_1[i]:.4f}")

rysuj_wykres(x_values_1, y_values_1, rozrzut_values_1, zakres_zmienności)

# 2. Inna wartość początkowa x w przedziale
x_values_2, y_values_2, rozrzut_values_2 = algorytm_1_1(rozrzut, wsp_przyrostu, 20, zakres_zmienności)
print("\nPrzykład 2 - zmiany po 20 iteracjach:")
print(f"Na początku:  y = {y_values_2[0]:.4f}, rozrzut = {rozrzut_values_2[0]:.4f}")
for i in range(1, 21):
    print(f"Po {i} iteracjach: y = {y_values_2[i]:.4f}, rozrzut = {rozrzut_values_2[i]:.4f}")

rysuj_wykres(x_values_2, y_values_2, rozrzut_values_2, zakres_zmienności)

# 3. Wylosowanie początkowego x z zakresu [15, 35] i rozrzut = 5
rozrzut = 5
zakres_zmienności = [15, 35]
x_values_3, y_values_3, rozrzut_values_3 = algorytm_1_1(rozrzut, wsp_przyrostu, 20, zakres_zmienności)
print("\nPrzykład 3 - x, y na początku i po 20 iteracjach:")
print(f"Na początku: x = {x_values_3[0]:.4f}, y = {y_values_3[0]:.4f}")
for i in range(1, 21):
    print(f"Po {i} iteracjach: x = {x_values_3[i]:.4f}, y = {y_values_3[i]:.4f}")

rysuj_wykres(x_values_3, y_values_3, rozrzut_values_3, zakres_zmienności)
