import numpy as np

from graf import generuj_graf, cechy_wierzcholkow, losuj_podatnosci
from atak import lista_sasiedztwa, symuluj_atak, czas_do_progu, wartosc_obronna


def zbior_treningowy(liczba_grafow=10, n=50, m=2, typy=("ba", "er", "ws"),
                     p=0.08, k=4, beta=0.5, liczba_atakow=60, ziarno=0,
                     postep=None):
    """Buduje zbior (X, y) z roznorodnej rodziny grafow.

    Dla kazdego typu topologii generujemy 'liczba_grafow' grafow, liczymy cechy
    wierzcholkow i ich wartosc obronna (etykieta). Trening na mieszance topologii
    sprawia, ze siec uczy sie strategii odpornej, a nie dopasowanej do jednego typu.
    """
    if isinstance(typy, str):
        typy = (typy,)
    X_lista, y_lista = [], []
    licznik = 0
    razem = liczba_grafow * len(typy)
    for typ in typy:
        for g in range(liczba_grafow):
            G = generuj_graf(typ=typ, n=n, m=m, p=p, k=k, ziarno=ziarno + g)
            w = losuj_podatnosci(G, ziarno=1000 + ziarno + g)
            sasiedzi = lista_sasiedztwa(G)
            nn = G.number_of_nodes()
            X_lista.append(cechy_wierzcholkow(G, w))
            y_lista.append(wartosc_obronna(sasiedzi, nn, w, beta=beta,
                                           liczba_atakow=liczba_atakow,
                                           ziarno=ziarno + g))
            licznik += 1
            if postep is not None:
                postep(licznik, razem)
    return np.vstack(X_lista), np.concatenate(y_lista)


def wybierz_obrone(strategia, G, budzet, X=None, predykcje=None, ziarno=0):
    """Zwraca zbior indeksow wezlow, ktore utwardzamy (h=1).

    strategia: 'losowa', 'stopien', 'posrednictwo', 'nn'.
    """
    import networkx as nx

    wezly = sorted(G.nodes())
    n = len(wezly)
    budzet = min(budzet, n)

    rng = np.random.default_rng(ziarno)

    def najlepsze(wartosci):
        # losowe rozstrzyganie remisow, zeby porownanie bylo uczciwe
        szum = rng.random(n) * 1e-9
        return set(np.argsort(np.asarray(wartosci) + szum)[::-1][:budzet].tolist())

    if strategia == "losowa":
        return set(rng.choice(n, size=budzet, replace=False).tolist())
    if strategia == "stopien":
        return najlepsze([G.degree(wezly[i]) for i in range(n)])
    if strategia == "posrednictwo":
        bc = nx.betweenness_centrality(G)
        return najlepsze([bc[wezly[i]] for i in range(n)])
    if strategia == "nn":
        return najlepsze(predykcje)
    raise ValueError("nieznana strategia: " + str(strategia))


def przebieg_strategii(G, bronione, podatnosc, beta=0.5, liczba_atakow=30,
                       max_krokow=200, ziarno=123):
    """Usrednione krzywe propagacji i metryki dla danej obrony.

    Zwraca slownik: srednia krzywa, czas do zarazenia polowy sieci,
    koncowy odsetek zainfekowanych.
    """
    sasiedzi = lista_sasiedztwa(G)
    n = G.number_of_nodes()
    h = np.zeros(n)
    for v in bronione:
        h[v] = 1.0

    rng = np.random.default_rng(ziarno)
    dozwolone = [v for v in range(n) if v not in bronione]
    starty = rng.choice(dozwolone, size=liczba_atakow,
                        replace=len(dozwolone) < liczba_atakow)

    krzywe = []
    czasy = []
    koncowe = []
    for start in starty:
        hist = symuluj_atak(sasiedzi, n, int(start), h, podatnosc, beta,
                            max_krokow, rng)
        krzywe.append(hist)
        czasy.append(czas_do_progu(hist, n, prog=0.5, kara=max_krokow))
        koncowe.append(hist[-1] / n)

    dlugosc = max(len(k) for k in krzywe)
    macierz = np.array([k + [k[-1]] * (dlugosc - len(k)) for k in krzywe], dtype=float)
    return {
        "krzywa": macierz.mean(axis=0),
        "czas_do_polowy": float(np.mean(czasy)),
        "koncowy_odsetek": float(np.mean(koncowe)),
    }
