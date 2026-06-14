import numpy as np


def lista_sasiedztwa(G):
    """Slownik wezel -> lista sasiadow (dla skierowanego: nastepnicy)."""
    if G.is_directed():
        return {v: list(G.successors(v)) for v in G.nodes()}
    return {v: list(G.neighbors(v)) for v in G.nodes()}


def symuluj_atak(sasiedzi, n, start, h, podatnosc, beta=0.5, max_krokow=200,
                 rng=None):
    """Symulacja propagacji ataku spacerem losowym (model SI).

    sasiedzi   - slownik wezel -> lista sasiadow
    n          - liczba wezlow
    start      - wezel poczatkowy (pacjent zero)
    h          - wektor utwardzenia w [0, 1]; h[i]=1 to wezel w pelni odporny
    podatnosc  - wektor podatnosci wezlow (parametr bezpieczenstwa)
    beta       - bazowa zarazliwosc
    Prawdopodobienstwo zarazenia wezla: beta * podatnosc[cel] * (1 - h[cel]).
    Zwraca historie liczby zainfekowanych w kolejnych krokach.
    """
    if rng is None:
        rng = np.random.default_rng()

    zainfekowany = np.zeros(n, dtype=bool)
    if h[start] >= 1.0:
        return [0]  # atak zablokowany juz na starcie
    zainfekowany[start] = True
    historia = [1]

    for _ in range(max_krokow):
        nowe = []
        for v in np.flatnonzero(zainfekowany):
            ss = sasiedzi[v]
            if not ss:
                continue
            cel = ss[rng.integers(len(ss))]  # krok spaceru losowego
            if not zainfekowany[cel]:
                if rng.random() < beta * podatnosc[cel] * (1.0 - h[cel]):
                    nowe.append(cel)
        for cel in nowe:
            zainfekowany[cel] = True
        historia.append(int(zainfekowany.sum()))

        # przerywamy gdy nie ma juz kogo zarazic
        if zainfekowany.sum() >= n - int((h >= 1.0).sum()):
            break
    return historia


def czas_do_progu(historia, n, prog=0.5, kara=None):
    """Numer kroku, w ktorym zainfekowano >= prog * n wezlow.

    Jesli prog nie zostal osiagniety, zwracamy 'kara' (domyslnie dlugosc historii),
    co premiuje obrone, ktora w ogole nie dopuscila do zarazenia polowy sieci.
    """
    cel = prog * n
    for i, liczba in enumerate(historia):
        if liczba >= cel:
            return i
    return kara if kara is not None else len(historia)


def wartosc_obronna(sasiedzi, n, podatnosc, beta=0.5, max_krokow=300,
                    liczba_atakow=60, ziarno=0):
    """Wartosc obronna kazdego wezla liczona metoda Monte Carlo.

    Symulujemy atak bez obrony i zapamietujemy drzewo propagacji (kto kogo zarazil).
    Wartoscia obronna wezla v jest sredni rozmiar jego poddrzewa infekcji, czyli
    liczba zarazen, ktore przeszly przez v. Utwardzenie takiego wezla blokuje
    najwiecej kolejnych zakazen. Poniewaz symulacja uwzglednia podatnosc, wartosc
    obronna premiuje wezly, ktore latwo sie zarazaja i duzo zarazaja dalej.
    Usredniamy po wielu losowych atakach.
    """
    rng = np.random.default_rng(ziarno)
    wartosci = np.zeros(n)
    for _ in range(liczba_atakow):
        start = int(rng.integers(n))
        zaraz = np.zeros(n, dtype=bool)
        zaraz[start] = True
        rodzic = np.full(n, -1)
        kolejnosc = [start]
        for _krok in range(max_krokow):
            nowe = []
            for v in np.flatnonzero(zaraz):
                ss = sasiedzi[v]
                if ss:
                    cel = ss[rng.integers(len(ss))]
                    if not zaraz[cel] and rng.random() < beta * podatnosc[cel]:
                        nowe.append((v, cel))
            for v, cel in nowe:
                if not zaraz[cel]:
                    zaraz[cel] = True
                    rodzic[cel] = v
                    kolejnosc.append(cel)
            if zaraz.all():
                break
        # rozmiary poddrzew: od najpozniej zarazonych w gore do korzenia
        rozmiar = np.zeros(n)
        for v in reversed(kolejnosc):
            rozmiar[v] += 1.0
            if rodzic[v] >= 0:
                rozmiar[rodzic[v]] += rozmiar[v]
        wartosci += rozmiar
    return wartosci / liczba_atakow
