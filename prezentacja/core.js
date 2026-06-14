// Logika modelu: generowanie grafow, symulacja ataku (spacer losowy) i siec
// neuronowa uczona spadkiem gradientu. Wszystko napisane od zera.

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function Rng(seed) {
  const f = mulberry32(seed >>> 0);
  return {
    random: f,
    int: (n) => Math.floor(f() * n),
    pick: (arr) => arr[Math.floor(f() * arr.length)],
  };
}

// ---------- generowanie grafow ----------

function najwiekszaSkladowa(n, adj) {
  const odw = new Array(n).fill(-1);
  let najlepsza = [];
  for (let s = 0; s < n; s++) {
    if (odw[s] !== -1) continue;
    const stos = [s]; odw[s] = s; const skl = [];
    while (stos.length) {
      const v = stos.pop(); skl.push(v);
      for (const u of adj[v]) if (odw[u] === -1) { odw[u] = s; stos.push(u); }
    }
    if (skl.length > najlepsza.length) najlepsza = skl;
  }
  // przenumerowanie wezlow
  const mapa = new Map();
  najlepsza.forEach((v, i) => mapa.set(v, i));
  const m = najlepsza.length;
  const nowe = Array.from({ length: m }, () => []);
  for (const v of najlepsza) {
    for (const u of adj[v]) if (mapa.has(u)) nowe[mapa.get(v)].push(mapa.get(u));
  }
  return { n: m, adj: nowe };
}

function generujGraf(typ, n, rng, opcje) {
  opcje = opcje || {};
  const m = opcje.m || 2, p = opcje.p || 0.05, k = opcje.k || 4;
  const zb = Array.from({ length: n }, () => new Set());
  const dodaj = (a, b) => { if (a !== b) { zb[a].add(b); zb[b].add(a); } };

  if (typ === 'ba') {
    const cele = [];
    for (let i = 0; i < m; i++) { for (let j = 0; j < m; j++) if (i !== j) dodaj(i, j); }
    for (let i = 0; i < m; i++) for (let r = 0; r < m; r++) cele.push(i);
    for (let v = m; v < n; v++) {
      const wybrani = new Set();
      while (wybrani.size < Math.min(m, v)) wybrani.add(rng.pick(cele));
      for (const u of wybrani) { dodaj(v, u); cele.push(u); cele.push(v); }
    }
  } else if (typ === 'er') {
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (rng.random() < p) dodaj(i, j);
  } else { // ws
    const kk = Math.floor(k / 2);
    for (let i = 0; i < n; i++) for (let d = 1; d <= kk; d++) dodaj(i, (i + d) % n);
    for (let i = 0; i < n; i++) for (let d = 1; d <= kk; d++) {
      if (rng.random() < p) {
        const j = (i + d) % n; zb[i].delete(j); zb[j].delete(i);
        let nowy = rng.int(n); let prob = 0;
        while ((nowy === i || zb[i].has(nowy)) && prob < 20) { nowy = rng.int(n); prob++; }
        dodaj(i, nowy);
      }
    }
  }
  const adj = zb.map((s) => Array.from(s));
  return najwiekszaSkladowa(n, adj);
}

function podatnosci(n, rng, frakcjaSerwerow) {
  frakcjaSerwerow = frakcjaSerwerow == null ? 0.4 : frakcjaSerwerow;
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = rng.random() < frakcjaSerwerow ? 0.25 : 1.0;
  return w;
}

// ---------- cechy wezlow ----------

function rozkladStacjonarny(n, adj, kroki) {
  kroki = kroki || 120;
  let pi = new Float64Array(n).fill(1 / n);
  const stopnie = adj.map((a) => a.length || 1);
  for (let t = 0; t < kroki; t++) {
    const nowe = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const wklad = pi[i] / stopnie[i];
      for (const j of adj[i]) nowe[j] += wklad;
    }
    pi = nowe;
  }
  let s = 0; for (let i = 0; i < n; i++) s += pi[i];
  for (let i = 0; i < n; i++) pi[i] /= s || 1;
  return pi;
}

function gronowanie(n, adj) {
  const sasiad = adj.map((a) => new Set(a));
  const c = new Float64Array(n);
  for (let v = 0; v < n; v++) {
    const ss = adj[v]; const d = ss.length;
    if (d < 2) { c[v] = 0; continue; }
    let polaczenia = 0;
    for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) if (sasiad[ss[i]].has(ss[j])) polaczenia++;
    c[v] = (2 * polaczenia) / (d * (d - 1));
  }
  return c;
}

function kRdzen(n, adj) {
  const stopien = adj.map((a) => a.length);
  const core = new Array(n).fill(0);
  const usuniety = new Array(n).fill(false);
  let poziom = 0;
  for (let cnt = 0; cnt < n; cnt++) {
    let min = Infinity, v = -1;
    for (let i = 0; i < n; i++) if (!usuniety[i] && stopien[i] < min) { min = stopien[i]; v = i; }
    if (v === -1) break;
    poziom = Math.max(poziom, min);
    core[v] = poziom;
    usuniety[v] = true;
    for (const u of adj[v]) if (!usuniety[u]) stopien[u]--;
  }
  return core;
}

const NAZWY_CECH = ['stopien', 'stacjonarny', 'gronowanie', 'rdzen', 'podatnosc'];

function cechy(graf, w) {
  const { n, adj } = graf;
  const stopnie = adj.map((a) => a.length);
  const sredni = Math.max(1e-9, stopnie.reduce((a, b) => a + b, 0) / n);
  const pi = rozkladStacjonarny(n, adj);
  const gron = gronowanie(n, adj);
  const core = kRdzen(n, adj);
  const maxCore = Math.max(1, ...core);
  const X = [];
  for (let v = 0; v < n; v++) {
    X.push([stopnie[v] / sredni, pi[v] * n, gron[v], core[v] / maxCore, w[v]]);
  }
  return X;
}

// ---------- symulacja ataku (spacer losowy, model SI) ----------

function symulujAtak(adj, n, start, h, w, beta, maxK, rng) {
  if (h[start] >= 1) return [0];
  const zaraz = new Uint8Array(n); zaraz[start] = 1;
  let liczba = 1; const hist = [1];
  let aktywne = [start];
  const odpornych = h.reduce((a, b) => a + (b >= 1 ? 1 : 0), 0);
  for (let krok = 0; krok < maxK; krok++) {
    const nowe = [];
    for (const v of aktywne) {
      const ss = adj[v];
      if (ss.length === 0) continue;
      const cel = ss[rng.int(ss.length)];
      if (!zaraz[cel] && rng.random() < beta * w[cel] * (1 - h[cel])) nowe.push(cel);
    }
    for (const c of nowe) if (!zaraz[c]) { zaraz[c] = 1; liczba++; aktywne.push(c); }
    hist.push(liczba);
    if (liczba >= n - odpornych) break;
  }
  return hist;
}

// drzewo propagacji -> wartosc obronna (rozmiar poddrzewa infekcji)
function wartoscObronna(adj, n, w, beta, maxK, ataki, rng) {
  const wart = new Float64Array(n);
  for (let a = 0; a < ataki; a++) {
    const start = rng.int(n);
    const zaraz = new Uint8Array(n); zaraz[start] = 1;
    const rodzic = new Int32Array(n).fill(-1);
    const kolejnosc = [start];
    let aktywne = [start];
    for (let krok = 0; krok < maxK; krok++) {
      const nowe = [];
      for (const v of aktywne) {
        const ss = adj[v];
        if (ss.length === 0) continue;
        const cel = ss[rng.int(ss.length)];
        if (!zaraz[cel] && rng.random() < beta * w[cel]) nowe.push([v, cel]);
      }
      let cosNowego = false;
      for (const [v, c] of nowe) if (!zaraz[c]) { zaraz[c] = 1; rodzic[c] = v; kolejnosc.push(c); aktywne.push(c); cosNowego = true; }
      if (!cosNowego && aktywne.length === 0) break;
      if (kolejnosc.length >= n) break;
    }
    const rozmiar = new Float64Array(n);
    for (let i = kolejnosc.length - 1; i >= 0; i--) {
      const v = kolejnosc[i]; rozmiar[v] += 1;
      if (rodzic[v] >= 0) rozmiar[rodzic[v]] += rozmiar[v];
    }
    for (let i = 0; i < n; i++) wart[i] += rozmiar[i];
  }
  for (let i = 0; i < n; i++) wart[i] /= ataki;
  return wart;
}

// ---------- standaryzacja i siec neuronowa ----------

function Standaryzacja() {
  return {
    srednia: null, odch: null,
    dopasuj(X) {
      const d = X[0].length, n = X.length;
      this.srednia = new Float64Array(d); this.odch = new Float64Array(d);
      for (const x of X) for (let j = 0; j < d; j++) this.srednia[j] += x[j];
      for (let j = 0; j < d; j++) this.srednia[j] /= n;
      for (const x of X) for (let j = 0; j < d; j++) this.odch[j] += (x[j] - this.srednia[j]) ** 2;
      for (let j = 0; j < d; j++) { this.odch[j] = Math.sqrt(this.odch[j] / n) || 1; }
      return this;
    },
    przeksztalc(X) {
      return X.map((x) => x.map((v, j) => (v - this.srednia[j]) / this.odch[j]));
    },
  };
}

function macierz(r, c, fn) {
  const M = []; for (let i = 0; i < r; i++) { const row = new Float64Array(c); for (let j = 0; j < c; j++) row[j] = fn ? fn() : 0; M.push(row); } return M;
}

function Siec(nIn, hidden, rng) {
  const rozmiary = [nIn, ...hidden, 1];
  const W = [], b = [];
  // pseudolosowe wagi z RNG (Gauss przez Box-Muller)
  const gauss = () => { let u = 0, v = 0; while (u === 0) u = rng.random(); while (v === 0) v = rng.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  for (let i = 0; i < rozmiary.length - 1; i++) {
    const skala = Math.sqrt(2 / rozmiary[i]);
    W.push(macierz(rozmiary[i], rozmiary[i + 1], () => gauss() * skala));
    b.push(new Float64Array(rozmiary[i + 1]));
  }
  function forward(X) {
    const akt = [X]; const zs = []; let a = X;
    for (let l = 0; l < W.length; l++) {
      const Wl = W[l], bl = b[l], wy = Wl[0].length;
      const z = a.map((row) => {
        const o = new Float64Array(wy);
        for (let j = 0; j < wy; j++) { let s = bl[j]; for (let i = 0; i < row.length; i++) s += row[i] * Wl[i][j]; o[j] = s; }
        return o;
      });
      zs.push(z);
      a = (l < W.length - 1) ? z.map((row) => row.map((v) => (v > 0 ? v : 0))) : z;
      akt.push(a);
    }
    return { akt, zs };
  }
  return {
    W, b, forward,
    predykcja(X) { return forward(X).akt[W.length].map((row) => row[0]); },
    trenuj(X, y, epoki, lr, batch, rngT) {
      const n = X.length; const hist = [];
      const idx = Array.from({ length: n }, (_, i) => i);
      for (let ep = 0; ep < epoki; ep++) {
        for (let i = n - 1; i > 0; i--) { const j = rngT.int(i + 1); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        for (let p = 0; p < n; p += batch) {
          const bi = idx.slice(p, p + batch);
          const Xb = bi.map((k) => X[k]); const yb = bi.map((k) => y[k]);
          const { akt, zs } = forward(Xb);
          const m = Xb.length;
          let dA = akt[W.length].map((row, r) => [ (2 / m) * (row[0] - yb[r]) ]);
          for (let l = W.length - 1; l >= 0; l--) {
            let dZ;
            if (l < W.length - 1) dZ = dA.map((row, r) => row.map((v, j) => (zs[l][r][j] > 0 ? v : 0)));
            else dZ = dA;
            const aPrev = akt[l];
            const wIn = W[l].length, wOut = W[l][0].length;
            const dW = macierz(wIn, wOut);
            const db = new Float64Array(wOut);
            for (let r = 0; r < m; r++) {
              for (let j = 0; j < wOut; j++) {
                const g = dZ[r][j]; db[j] += g;
                const ar = aPrev[r];
                for (let i = 0; i < wIn; i++) dW[i][j] += ar[i] * g;
              }
            }
            const dAprev = aPrev.map(() => new Float64Array(wIn));
            for (let r = 0; r < m; r++) for (let i = 0; i < wIn; i++) { let s = 0; for (let j = 0; j < wOut; j++) s += dZ[r][j] * W[l][i][j]; dAprev[r][i] = s; }
            for (let i = 0; i < wIn; i++) for (let j = 0; j < wOut; j++) W[l][i][j] -= lr * dW[i][j];
            for (let j = 0; j < wOut; j++) b[l][j] -= lr * db[j];
            dA = dAprev.map((row) => Array.from(row));
          }
        }
        const pred = this.predykcja(X);
        let mse = 0; for (let i = 0; i < n; i++) mse += (pred[i] - y[i]) ** 2;
        hist.push(mse / n);
      }
      return hist;
    },
  };
}

// ---------- strategie obrony ----------

function topK(wartosci, k, rng) {
  const n = wartosci.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => (wartosci[b] + rng.random() * 1e-9) - (wartosci[a] + rng.random() * 1e-9));
  return new Set(idx.slice(0, k));
}

function wybierzObrone(strat, graf, w, budzet, pred, rng) {
  const { n, adj } = graf;
  budzet = Math.min(budzet, n);
  if (strat === 'brak') return new Set();
  if (strat === 'losowa') {
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = rng.int(i + 1); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    return new Set(idx.slice(0, budzet));
  }
  if (strat === 'stopien') return topK(adj.map((a) => a.length), budzet, rng);
  if (strat === 'nn') return topK(pred, budzet, rng);
  throw new Error('nieznana strategia ' + strat);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    Rng, generujGraf, podatnosci, cechy, NAZWY_CECH, symulujAtak,
    wartoscObronna, Standaryzacja, Siec, wybierzObrone, topK, rozkladStacjonarny,
  };
}
