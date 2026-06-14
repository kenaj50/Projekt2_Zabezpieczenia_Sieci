// Interfejs prezentacji: trenuje model, generuje sieci i animuje propagacje ataku.

const S = {
  net: null, skaler: null, waznosc: null, strataHist: null,
  graf: null, w: null, pos: null, pred: null,
  obrony: null, start: null, czasyNone: null, czasyNn: null,
  krzywe: null, czasyPolowa: null, anim: null,
};

const $ = (id) => document.getElementById(id);
const KOLORY = { brak: '#ef4444', losowa: '#f59e0b', stopien: '#8b5cf6', nn: '#22c55e' };
const ETYK = { brak: 'Brak', losowa: 'Losowa', stopien: 'Stopień', nn: 'Sieć neuronowa' };

// ---------- trening modelu ----------

function trenujModel() {
  const rng = Rng(2024);
  let X = [], y = [];
  for (const typ of ['ba', 'er', 'ws']) {
    for (let g = 0; g < 6; g++) {
      const graf = generujGraf(typ, 45, rng, { m: 2, p: 0.09, k: 4 });
      const w = podatnosci(graf.n, rng);
      const c = cechy(graf, w);
      const wart = wartoscObronna(graf.adj, graf.n, w, 0.45, 250, 30, rng);
      for (let v = 0; v < graf.n; v++) { X.push(c[v]); y.push(wart[v]); }
    }
  }
  // standaryzacja wejscia oraz celu (stabilniejszy spadek gradientu)
  const my = y.reduce((a, b) => a + b, 0) / y.length;
  const sy = Math.sqrt(y.reduce((a, b) => a + (b - my) ** 2, 0) / y.length) || 1;
  const yz = y.map((v) => (v - my) / sy);

  S.skaler = Standaryzacja().dopasuj(X);
  const Xs = S.skaler.przeksztalc(X);
  S.net = Siec(X[0].length, [24, 12], Rng(1));
  S.strataHist = S.net.trenuj(Xs, yz, 220, 0.01, 32, Rng(2));
  S.waznosc = waznoscCech(S.net, Xs, yz);
}

function waznoscCech(net, X, y) {
  const rng = Rng(7);
  const baza = mse(net.predykcja(X), y);
  const out = {};
  for (let j = 0; j < NAZWY_CECH.length; j++) {
    let suma = 0;
    for (let r = 0; r < 6; r++) {
      const Xp = X.map((row) => row.slice());
      const kol = Xp.map((row) => row[j]);
      for (let i = kol.length - 1; i > 0; i--) { const k = rng.int(i + 1); const t = kol[i]; kol[i] = kol[k]; kol[k] = t; }
      Xp.forEach((row, i) => { row[j] = kol[i]; });
      suma += mse(net.predykcja(Xp), y) - baza;
    }
    out[NAZWY_CECH[j]] = suma / 6;
  }
  return out;
}

function mse(p, y) { let s = 0; for (let i = 0; i < y.length; i++) s += (p[i] - y[i]) ** 2; return s / y.length; }

// ---------- uklad grafu (force-directed) ----------

function uklad(graf, iters) {
  iters = iters || 170;
  const { n, adj } = graf;
  const rng = Rng(99);
  let pos = Array.from({ length: n }, () => [rng.random(), rng.random()]);
  const k = Math.sqrt(1 / n);
  let temp = 0.12;
  const edges = [];
  for (let i = 0; i < n; i++) for (const j of adj[i]) if (i < j) edges.push([i, j]);
  for (let it = 0; it < iters; it++) {
    const disp = Array.from({ length: n }, () => [0, 0]);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
      let d = Math.hypot(dx, dy) || 1e-4;
      const f = (k * k) / d, ux = dx / d, uy = dy / d;
      disp[i][0] += ux * f; disp[i][1] += uy * f;
      disp[j][0] -= ux * f; disp[j][1] -= uy * f;
    }
    for (const [i, j] of edges) {
      let dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
      let d = Math.hypot(dx, dy) || 1e-4;
      const f = (d * d) / k, ux = dx / d, uy = dy / d;
      disp[i][0] -= ux * f; disp[i][1] -= uy * f;
      disp[j][0] += ux * f; disp[j][1] += uy * f;
    }
    for (let i = 0; i < n; i++) {
      let d = Math.hypot(disp[i][0], disp[i][1]) || 1e-4;
      pos[i][0] += (disp[i][0] / d) * Math.min(d, temp);
      pos[i][1] += (disp[i][1] / d) * Math.min(d, temp);
    }
    temp *= 0.975;
  }
  // normalizacja do [0,1]
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  for (const [x, y] of pos) { minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
  return pos.map(([x, y]) => [(x - minx) / (maxx - minx || 1), (y - miny) / (maxy - miny || 1)]);
}

// ---------- generowanie sieci do pokazu ----------

function nowaSiec() {
  const typ = $('topo').value;
  const n = +$('n').value;
  const ziarno = (Math.random() * 1e9) | 0;
  S.graf = generujGraf(typ, n, Rng(ziarno), { m: 2, p: 0.05, k: 4 });
  S.w = podatnosci(S.graf.n, Rng(ziarno + 1));
  S.pred = S.net.predykcja(S.skaler.przeksztalc(cechy(S.graf, S.w)));
  S.pos = uklad(S.graf);
  obliczObrony();
  S.czasyNone = null; S.czasyNn = null; S.krzywe = null;
  rysujPlansze(0); rysujPlansze(0, true);
  rysujWykres(0);
  $('verdict').innerHTML = 'Naciśnij <b>ATAK!</b>';
  $('bars').innerHTML = '';
  $('stat-none').textContent = 'zainfekowane: 0';
  $('stat-ai').textContent = 'zainfekowane: 0';
}

function obliczObrony() {
  const B = Math.min(+$('budget').value, S.graf.n);
  S.obrony = {};
  for (const strat of ['brak', 'losowa', 'stopien', 'nn'])
    S.obrony[strat] = wybierzObrone(strat, S.graf, S.w, B, S.pred, Rng(7));
}

// ---------- przebieg z czasami infekcji ----------

function przebiegCzasy(adj, n, start, h, w, beta, maxK, rng) {
  const czas = new Int32Array(n).fill(-1);
  if (h[start] >= 1) return czas;
  czas[start] = 0;
  const zaraz = new Uint8Array(n); zaraz[start] = 1;
  let aktywne = [start];
  const odp = h.reduce((a, b) => a + (b >= 1 ? 1 : 0), 0);
  let liczba = 1;
  for (let krok = 1; krok <= maxK; krok++) {
    const nowe = [];
    for (const v of aktywne) {
      const ss = adj[v];
      if (ss.length === 0) continue;
      const cel = ss[rng.int(ss.length)];
      if (!zaraz[cel] && rng.random() < beta * w[cel] * (1 - h[cel])) nowe.push(cel);
    }
    for (const c of nowe) if (!zaraz[c]) { zaraz[c] = 1; czas[c] = krok; aktywne.push(c); liczba++; }
    if (liczba >= n - odp) break;
  }
  return czas;
}

function sredniaKrzywa(strat, beta) {
  const { adj, n } = S.graf;
  const h = new Float64Array(n);
  for (const v of S.obrony[strat]) h[v] = 1;
  const dozwolone = []; for (let v = 0; v < n; v++) if (!S.obrony[strat].has(v)) dozwolone.push(v);
  const rng = Rng(123);
  const krzywe = [];
  for (let a = 0; a < 14; a++) {
    const start = dozwolone[rng.int(dozwolone.length)];
    krzywe.push(symulujAtak(adj, n, start, h, S.w, beta, 200, rng).map((x) => x / n));
  }
  const dl = Math.max(...krzywe.map((c) => c.length));
  const sred = new Float64Array(dl);
  for (const c of krzywe) for (let i = 0; i < dl; i++) sred[i] += (i < c.length ? c[i] : c[c.length - 1]);
  for (let i = 0; i < dl; i++) sred[i] /= krzywe.length;
  let polowa = dl; for (let i = 0; i < dl; i++) if (sred[i] >= 0.5) { polowa = i; break; }
  return { krzywa: Array.from(sred), polowa };
}

function przygotujAtak() {
  const beta = +$('beta').value;
  const { adj, n } = S.graf;
  // wspolny wezel startowy (poza zaporami NN), zeby plansze byly porownywalne
  const typStart = $('start').value;
  let kandydaci = []; for (let v = 0; v < n; v++) if (!S.obrony.nn.has(v)) kandydaci.push(v);
  if (typStart === 'centralny') {
    kandydaci.sort((a, b) => adj[b].length - adj[a].length);
    S.start = kandydaci[0];
  } else {
    S.start = kandydaci[(Math.random() * kandydaci.length) | 0];
  }
  const ziarnoWsp = (Math.random() * 1e9) | 0;
  const hNone = new Float64Array(n);
  const hNn = new Float64Array(n); for (const v of S.obrony.nn) hNn[v] = 1;
  S.czasyNone = przebiegCzasy(adj, n, S.start, hNone, S.w, beta, 200, Rng(ziarnoWsp));
  S.czasyNn = przebiegCzasy(adj, n, S.start, hNn, S.w, beta, 200, Rng(ziarnoWsp));

  S.krzywe = {}; S.czasyPolowa = {};
  for (const strat of ['brak', 'losowa', 'stopien', 'nn']) {
    const r = sredniaKrzywa(strat, beta);
    S.krzywe[strat] = r.krzywa; S.czasyPolowa[strat] = r.polowa;
  }
}

// ---------- rysowanie planszy ----------

function lerpKolor(t) {
  // niebieski (niska podatnosc) -> pomaranczowy (wysoka)
  const a = [56, 189, 248], b = [251, 146, 60];
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function rysujPlansze(krok, zObrona) {
  const cv = zObrona ? $('cv-ai') : $('cv-none');
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, pad = 26;
  ctx.clearRect(0, 0, W, H);
  if (!S.graf) return;
  const { n, adj } = S.graf;
  const px = (i) => pad + S.pos[i][0] * (W - 2 * pad);
  const py = (i) => pad + S.pos[i][1] * (H - 2 * pad);
  const czasy = zObrona ? S.czasyNn : S.czasyNone;
  const obrona = zObrona ? S.obrony.nn : new Set();

  ctx.strokeStyle = 'rgba(120,140,180,0.13)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < n; i++) for (const j of adj[i]) if (i < j) { ctx.moveTo(px(i), py(i)); ctx.lineTo(px(j), py(j)); }
  ctx.stroke();

  const sred = adj.reduce((a, x) => a + x.length, 0) / n || 1;
  for (let i = 0; i < n; i++) {
    const r = 3 + 4 * Math.min(2.5, adj[i].length / sred);
    const zar = czasy && czasy[i] >= 0 && czasy[i] <= krok;
    const x = px(i), y = py(i);
    if (obrona.has(i)) {
      ctx.fillStyle = '#0d2a16';
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
      ctx.shadowColor = '#22c55e'; ctx.shadowBlur = 10;
      ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.arc(x, y, r + 1.5, 0, 7); ctx.stroke();
      ctx.shadowBlur = 0;
    } else if (zar) {
      ctx.shadowColor = '#ef4444'; ctx.shadowBlur = 12;
      ctx.fillStyle = '#ef4444';
      ctx.beginPath(); ctx.arc(x, y, r + 0.5, 0, 7); ctx.fill();
      ctx.shadowBlur = 0;
    } else {
      ctx.fillStyle = lerpKolor((S.w[i] - 0.25) / 0.75);
      ctx.globalAlpha = 0.9;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
      ctx.globalAlpha = 1;
    }
  }
}

// ---------- rysowanie wykresu ----------

function rysujWykres(krok) {
  const cv = $('cv-chart'); const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, L = 46, B = 32;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#243355'; ctx.fillStyle = '#7e8db0'; ctx.font = '11px sans-serif'; ctx.lineWidth = 1;
  // osie
  ctx.beginPath(); ctx.moveTo(L, 10); ctx.lineTo(L, H - B); ctx.lineTo(W - 10, H - B); ctx.stroke();
  for (let p = 0; p <= 100; p += 25) {
    const yy = (H - B) - (p / 100) * (H - B - 10);
    ctx.strokeStyle = 'rgba(120,140,180,0.12)';
    ctx.beginPath(); ctx.moveTo(L, yy); ctx.lineTo(W - 10, yy); ctx.stroke();
    ctx.fillText(p + '%', 14, yy + 4);
  }
  ctx.fillText('krok ataku', W - 78, H - 10);
  if (!S.krzywe) return;
  const maxLen = Math.max(...Object.values(S.krzywe).map((c) => c.length));
  const sx = (i) => L + (i / Math.max(1, maxLen - 1)) * (W - 10 - L);
  const sy = (v) => (H - B) - v * (H - B - 10);
  for (const strat of ['brak', 'losowa', 'stopien', 'nn']) {
    const c = S.krzywe[strat];
    ctx.strokeStyle = KOLORY[strat]; ctx.lineWidth = strat === 'nn' ? 3 : 2;
    ctx.beginPath();
    const lim = Math.min(krok, c.length - 1);
    for (let i = 0; i <= lim; i++) { const X = sx(i), Y = sy(c[i]); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
    ctx.stroke();
  }
}

// ---------- statystyki / werdykt ----------

function pokazWynik() {
  const cz = S.czasyPolowa;
  const maxV = Math.max(...Object.values(cz), 1);
  let html = '';
  for (const strat of ['brak', 'losowa', 'stopien', 'nn']) {
    const v = cz[strat];
    html += `<div class="bar-row"><span class="name">${ETYK[strat]}</span>`
      + `<span class="bar-track"><span class="bar-fill" style="width:${(v / maxV * 100).toFixed(0)}%;background:${KOLORY[strat]}"></span></span>`
      + `<span class="val">${v.toFixed(0)}</span></div>`;
  }
  $('bars').innerHTML = html;
  const najlepszaHeur = Math.max(cz.stopien, cz.losowa);
  const nn = cz.nn;
  let txt;
  if (nn >= najlepszaHeur && nn > cz.losowa * 1.3)
    txt = `Obrona sieci neuronowej opóźniła atak najmocniej — ${(nn / Math.max(1, cz.losowa)).toFixed(1)}× dłużej niż obrona losowa.`;
  else if (nn > cz.losowa * 1.3)
    txt = `Sieć neuronowa dorównuje najlepszej heurystyce i jest ${(nn / Math.max(1, cz.losowa)).toFixed(1)}× lepsza od losowej.`;
  else txt = 'Sprawdź różne topologie i budżety — różnica zależy od struktury sieci.';
  $('verdict').innerHTML = txt;
}

// ---------- animacja ----------

function animuj() {
  if (S.anim) cancelAnimationFrame(S.anim);
  przygotujAtak();
  const maxCzas = Math.max(
    Math.max(...Array.from(S.czasyNone)),
    Math.max(...Array.from(S.czasyNn)),
    Math.max(...Object.values(S.krzywe).map((c) => c.length))
  );
  const T = maxCzas + 2;
  const krokNaRamke = Math.max(1, Math.ceil(T / 90));
  let k = 0, ostatni = 0;
  $('atak').disabled = true;
  function ramka(ts) {
    if (ts - ostatni > 45) {
      ostatni = ts;
      rysujPlansze(k, false); rysujPlansze(k, true);
      rysujWykres(k);
      const zN = liczZar(S.czasyNone, k), zNn = liczZar(S.czasyNn, k);
      $('stat-none').textContent = `zainfekowane: ${zN} / ${S.graf.n}`;
      $('stat-ai').textContent = `zainfekowane: ${zNn} / ${S.graf.n} (zapory: ${S.obrony.nn.size})`;
      k += krokNaRamke;
    }
    if (k <= T) S.anim = requestAnimationFrame(ramka);
    else { $('atak').disabled = false; pokazWynik(); }
  }
  S.anim = requestAnimationFrame(ramka);
}

function liczZar(czasy, k) { let c = 0; for (let i = 0; i < czasy.length; i++) if (czasy[i] >= 0 && czasy[i] <= k) c++; return c; }

// ---------- wykres straty i waznosc cech ----------

function rysujStrate() {
  const cv = $('cv-loss'); const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const h = S.strataHist; const mx = Math.max(...h), mn = Math.min(...h);
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < h.length; i++) {
    const x = (i / (h.length - 1)) * (W - 8) + 4;
    const y = H - 6 - ((h[i] - mn) / (mx - mn || 1)) * (H - 14);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = '#7e8db0'; ctx.font = '10px sans-serif';
  ctx.fillText('strata (MSE) →', 6, 12);
}

function pokazWaznosc() {
  const naz = { stopien: 'stopień', stacjonarny: 'spacer losowy', gronowanie: 'gronowanie', rdzen: 'k-rdzeń', podatnosc: 'podatność' };
  const pos = Object.entries(S.waznosc).sort((a, b) => b[1] - a[1]);
  const mx = Math.max(...pos.map((e) => e[1]), 1e-6);
  let html = '';
  for (const [k, v] of pos) {
    const proc = Math.max(2, (v / mx) * 100);
    const wazne = (k === 'podatnosc');
    html += `<div>${wazne ? '<b>' : ''}${naz[k] || k}${wazne ? '</b>' : ''}: `
      + `<span style="display:inline-block;height:8px;width:${proc}px;background:${wazne ? '#ef4444' : '#38bdf8'};border-radius:3px;vertical-align:middle"></span></div>`;
  }
  $('feat').innerHTML = html;
}

// ---------- start ----------

function start() {
  trenujModel();
  rysujStrate();
  pokazWaznosc();
  nowaSiec();
  $('overlay').classList.add('hidden');
}

$('n').addEventListener('input', () => { $('nVal').textContent = $('n').value; });
$('budget').addEventListener('input', () => { $('bVal').textContent = $('budget').value; if (S.graf) { obliczObrony(); rysujPlansze(0, true); } });
$('beta').addEventListener('input', () => { $('betaVal').textContent = (+$('beta').value).toFixed(2); });
$('nowa').addEventListener('click', nowaSiec);
$('topo').addEventListener('change', nowaSiec);
$('atak').addEventListener('click', animuj);

window.addEventListener('load', () => { setTimeout(start, 60); });
