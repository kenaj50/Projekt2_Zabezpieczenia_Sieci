# Projekt 2 — Zabezpieczenia sieci z wykorzystaniem sieci neuronowej

Matematyczne fundamenty informatyki, projekt nr 8.

Sieć komputerów modelujemy jako graf. Atak to spacer losowy — zainfekowany węzeł
próbuje zarazić losowego sąsiada. Każdy węzeł ma podatność (serwer vs stacja).
Uczymy sieć neuronową, która wskazuje, które węzły utwardzić, żeby spowolnić
propagację. Porównujemy to z obroną losową i heurystykami (stopień, pośrednictwo).

## Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab Zabezpieczenia_Sieci_NN.ipynb
```

Prezentacja interaktywna (otwórz w przeglądarce, bez serwera):

```
prezentacja/index.html
```

## Struktura

- `src/` — kod: graf, symulacja ataku, sieć neuronowa (numpy), dane treningowe
- `Zabezpieczenia_Sieci_NN.ipynb` — notebook z całą analizą i wynikami
- `prezentacja/` — demo z animacją ataku

## Wymagania

Python 3.10+, pakiety z `requirements.txt` (numpy, networkx, matplotlib, jupyter, ipywidgets).
