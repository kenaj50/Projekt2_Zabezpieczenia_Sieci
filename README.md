# Zabezpieczenia sieci z wykorzystaniem sieci neuronowej

Matematyczne fundamenty informatyki.

Sieć komputerów traktujemy jako graf. Atak to spacer losowy: zainfekowany węzeł
próbuje zarazić losowego sąsiada; każdy węzeł ma podatność (serwer vs stacja).
Uczymy sieć neuronową, która wskazuje, które węzły utwardzić, żeby spowolnić
propagację i na końcu orównujemy to z obroną losową i heurystykami (stopień, pośrednictwo).

## Uruchomienie


```bash
cd ".."
python3 -m venv .venv          # tylko raz, jesli jeszcze nie ma
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab Zabezpieczenia_Sieci_NN.ipynb
```
Lub po prostu uruchomic lokalnie Zabezpieczenia_Sieci_NN.ipynb 


Prezentacja (w przeglądarce):

```
prezentacja/index.html
```

## Pliki

- `src/` - graf, atak, sieć neuronowa, dane
- `Zabezpieczenia_Sieci_NN.ipynb` - notebook z wynikami
- `prezentacja/` - demo z animacją

## Wymagania

Python 3.10+, pakiety z `requirements.txt`.
