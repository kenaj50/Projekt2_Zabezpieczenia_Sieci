import numpy as np


def lista_sasiedztwa(G):
    """Slownik wezel -> sasiedzi."""
    if G.is_directed():
        return {v: list(G.successors(v)) for v in G.nodes()}
    return {v: list(G.neighbors(v)) for v in G.nodes()}


def symuluj_atak(sasiedzi, n, start, h, podatnosc, beta=0.5, max_krokow=200,
                 rng=None):
    """Symulacja ataku spacerem losowym (model SI)."""
    if rng is None:
        rng = np.random.default_rng()

    zainfekowany = np.zeros(n, dtype=bool)
    if h[start] >= 1.0:
        return [0]
    zainfekowany[start] = True
    historia = [1]

    for _ in range(max_krokow):
        nowe = []
        for v in np.flatnonzero(zainfekowany):
            ss = sasiedzi[v]
            if not ss:
                continue
            cel = ss[rng.integers(len(ss))]
            if not zainfekowany[cel]:
                if rng.random() < beta * podatnosc[cel] * (1.0 - h[cel]):
                    nowe.append(cel)
        for cel in nowe:
            zainfekowany[cel] = True
        historia.append(int(zainfekowany.sum()))

        if zainfekowany.sum() >= n - int((h >= 1.0).sum()):
            break
    return historia


def czas_do_progu(historia, n, prog=0.5, kara=None):
    """Krok, w ktorym zainfekowano prog * n wezlow."""
    cel = prog * n
    for i, liczba in enumerate(historia):
        if liczba >= cel:
            return i
    return kara if kara is not None else len(historia)


def wartosc_obronna(sasiedzi, n, podatnosc, beta=0.5, max_krokow=300,
                    liczba_atakow=60, ziarno=0):
    """Sredni rozmiar poddrzewa infekcji dla kazdego wezla (Monte Carlo)."""
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
        rozmiar = np.zeros(n)
        for v in reversed(kolejnosc):
            rozmiar[v] += 1.0
            if rodzic[v] >= 0:
                rozmiar[rodzic[v]] += rozmiar[v]
        wartosci += rozmiar
    return wartosci / liczba_atakow
