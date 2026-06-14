import numpy as np


class Standaryzacja:
    """Standaryzacja cech (z-score). Liczona na zbiorze treningowym."""

    def __init__(self):
        self.srednia = None
        self.odchylenie = None

    def dopasuj(self, X):
        self.srednia = X.mean(axis=0)
        self.odchylenie = X.std(axis=0)
        self.odchylenie[self.odchylenie == 0] = 1.0
        return self

    def przeksztalc(self, X):
        return (X - self.srednia) / self.odchylenie

    def dopasuj_przeksztalc(self, X):
        return self.dopasuj(X).przeksztalc(X)


class SiecNeuronowa:
    """Perceptron wielowarstwowy uczony spadkiem gradientu (od zera).

    Warstwy ukryte z aktywacja ReLU, wyjscie liniowe (regresja).
    Strata: blad sredniokwadratowy. Gradienty z propagacji wstecznej.
    """

    def __init__(self, n_wejsc, ukryte=(32, 16), ziarno=0):
        rng = np.random.default_rng(ziarno)
        rozmiary = [n_wejsc, *ukryte, 1]
        self.W = []
        self.b = []
        for i in range(len(rozmiary) - 1):
            # inicjalizacja He dla warstw z ReLU
            skala = np.sqrt(2.0 / rozmiary[i])
            self.W.append(rng.normal(0.0, skala, (rozmiary[i], rozmiary[i + 1])))
            self.b.append(np.zeros(rozmiary[i + 1]))

    def _forward(self, X):
        aktywacje = [X]
        z_lista = []
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            z_lista.append(z)
            if i < len(self.W) - 1:
                a = np.maximum(0.0, z)  # ReLU
            else:
                a = z  # wyjscie liniowe
            aktywacje.append(a)
        return aktywacje, z_lista

    def predykcja(self, X):
        aktywacje, _ = self._forward(X)
        return aktywacje[-1].ravel()

    def trenuj(self, X, y, epoki=400, lr=0.01, batch=32, ziarno=0):
        rng = np.random.default_rng(ziarno)
        y = y.reshape(-1, 1)
        n = X.shape[0]
        historia_straty = []

        for _ in range(epoki):
            kolejnosc = rng.permutation(n)
            for poczatek in range(0, n, batch):
                idx = kolejnosc[poczatek:poczatek + batch]
                Xb, yb = X[idx], y[idx]
                aktywacje, z_lista = self._forward(Xb)
                yhat = aktywacje[-1]

                m = Xb.shape[0]
                # pochodna straty MSE po wyjsciu
                dA = (2.0 / m) * (yhat - yb)
                # propagacja wsteczna
                for i in reversed(range(len(self.W))):
                    if i < len(self.W) - 1:
                        dZ = dA * (z_lista[i] > 0)  # pochodna ReLU
                    else:
                        dZ = dA
                    dW = aktywacje[i].T @ dZ
                    db = dZ.sum(axis=0)
                    dA = dZ @ self.W[i].T
                    self.W[i] -= lr * dW
                    self.b[i] -= lr * db

            yhat_all = self.predykcja(X).reshape(-1, 1)
            historia_straty.append(float(np.mean((yhat_all - y) ** 2)))
        return historia_straty


def waznosc_cech(siec, X, y, nazwy, powtorzenia=10, ziarno=0):
    """Waznosc cech metoda permutacji.

    Mieszamy po kolei kazda ceche i mierzymy, o ile wzrasta blad. Im wiekszy
    wzrost, tym wazniejsza cecha dla modelu.
    """
    rng = np.random.default_rng(ziarno)
    bazowy = np.mean((siec.predykcja(X) - y) ** 2)
    wynik = {}
    for j, nazwa in enumerate(nazwy):
        wzrosty = []
        for _ in range(powtorzenia):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            wzrosty.append(np.mean((siec.predykcja(Xp) - y) ** 2) - bazowy)
        wynik[nazwa] = float(np.mean(wzrosty))
    return wynik
