import numpy as np
import networkx as nx


def generuj_graf(typ="ba", n=60, m=2, p=0.08, k=4, skierowany=False, ziarno=None):
    """Graf typu ba, er lub ws. Opcjonalnie skierowany."""
    if typ == "ba":
        G = nx.barabasi_albert_graph(n, m, seed=ziarno)
    elif typ == "er":
        G = nx.erdos_renyi_graph(n, p, seed=ziarno)
    elif typ == "ws":
        G = nx.watts_strogatz_graph(n, k, p, seed=ziarno)
    else:
        raise ValueError("nieznany typ grafu: " + str(typ))

    # najwieksza skladowa spojna
    if not nx.is_connected(G):
        najwieksza = max(nx.connected_components(G), key=len)
        G = G.subgraph(najwieksza).copy()
    G = nx.convert_node_labels_to_integers(G)

    if skierowany:
        rng = np.random.default_rng(ziarno)
        D = nx.DiGraph()
        D.add_nodes_from(G.nodes())
        for u, v in G.edges():
            if rng.random() < 0.5:
                D.add_edge(u, v)
            else:
                D.add_edge(v, u)
        return D
    return G


def macierz_sasiedztwa(G):
    return nx.to_numpy_array(G, nodelist=sorted(G.nodes()))


def macierz_przejscia(G):
    """Macierz przejscia spaceru losowego (normalizacja wierszami)."""
    A = macierz_sasiedztwa(G)
    stopnie = A.sum(axis=1, keepdims=True)
    stopnie[stopnie == 0] = 1.0
    return A / stopnie


def rozklad_stacjonarny(G, kroki=200, tol=1e-10):
    """Rozklad stacjonarny spaceru losowego (metoda potegowa)."""
    P = macierz_przejscia(G)
    n = P.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(kroki):
        nowe = pi @ P
        if np.abs(nowe - pi).sum() < tol:
            pi = nowe
            break
        pi = nowe
    suma = pi.sum()
    return pi / suma if suma > 0 else pi


NAZWY_CECH = [
    "stopien",
    "posrednictwo",
    "bliskosc",
    "stacjonarny",
    "gronowanie",
    "rdzen",
    "podatnosc",
]


def losuj_podatnosci(G, frakcja_serwerow=0.4, pod_serwer=0.25, pod_stacja=1.0,
                     ziarno=None):
    """Losuje podatnosc wezlow: serwery niska, stacje wysoka."""
    rng = np.random.default_rng(ziarno)
    n = G.number_of_nodes()
    serwer = rng.random(n) < frakcja_serwerow
    return np.where(serwer, pod_serwer, pod_stacja)


def cechy_wierzcholkow(G, podatnosc):
    """Cechy wezlow: stopien, centralnosci, podatnosc itd."""
    wezly = sorted(G.nodes())
    n = len(wezly)
    stopnie = dict(G.degree())
    sredni_stopien = max(1e-9, np.mean([stopnie[v] for v in wezly]))
    posrednictwo = nx.betweenness_centrality(G)
    bliskosc = nx.closeness_centrality(G)
    stacjonarny = rozklad_stacjonarny(G)

    H = G.to_undirected() if G.is_directed() else G
    gronowanie = nx.clustering(H)
    rdzen = nx.core_number(H)
    max_rdzen = max(1, max(rdzen.values()))

    X = np.zeros((n, len(NAZWY_CECH)))
    for i, v in enumerate(wezly):
        X[i] = [
            stopnie[v] / sredni_stopien,
            posrednictwo[v],
            bliskosc[v],
            stacjonarny[i] * n,
            gronowanie[v],
            rdzen[v] / max_rdzen,
            podatnosc[i],
        ]
    return X
