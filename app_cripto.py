"""
App de análise de entradas em criptoativos (gráfico + volume + modelo ML).
Uso: python app_cripto.py APT --intervalo 1h   (gera grafico.png)
Colab: use o notebook. Estudo educacional, não é recomendação de investimento.
"""
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

HORIZONTE = 6        # candles à frente (6 x 1h = 6 horas)
LIMIAR = 0.01        # alta mínima de 1%
CUSTO = 0.001        # 0,1% por lado (taxa + slippage aproximado)
PROB_ENTRADA = 0.60  # probabilidade mínima para entrar

# ----------------------------------------------------------------- dados
def _get(url, params, tentativas=3):
    import requests
    for i in range(tentativas):
        r = requests.get(url, params=params, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 429:
            time.sleep(2 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("limite de requisições (429)")


def _ohlcv(rows, cols, ordem):
    """rows -> DataFrame OHLCV indexado por data (ms). cols: posição dos campos."""
    a = np.array(rows, dtype=float)
    df = pd.DataFrame({"Open": a[:, ordem[1]], "High": a[:, ordem[2]],
                       "Low": a[:, ordem[3]], "Close": a[:, ordem[4]],
                       "Volume": a[:, ordem[5]]},
                      index=pd.to_datetime(a[:, ordem[0]] * cols, unit="ms"))
    df = df[~df.index.duplicated()].sort_index()
    return df


def de_binance(simbolo, intervalo, paginas, base="https://api.binance.com"):
    rows, fim = [], None
    for _ in range(paginas):
        p = {"symbol": simbolo, "interval": intervalo, "limit": 1000}
        if fim:
            p["endTime"] = fim
        k = _get(base + "/api/v3/klines", p)
        if not k:
            break
        rows = k + rows
        fim = k[0][0] - 1
        if len(k) < 1000:
            break
    a = np.array(rows, dtype=float)
    df = pd.DataFrame({"Open": a[:, 1], "High": a[:, 2], "Low": a[:, 3],
                       "Close": a[:, 4], "Volume": a[:, 5],
                       "TakerRatio": a[:, 9] / (a[:, 5] + 1e-9)},
                      index=pd.to_datetime(a[:, 0], unit="ms"))
    return df[~df.index.duplicated()].sort_index()


def de_binance_vision(simbolo, intervalo, paginas):
    return de_binance(simbolo, intervalo, paginas, "https://data-api.binance.vision")


def de_bybit(simbolo, intervalo, paginas):
    mp = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
          "4h": "240", "1d": "D"}
    rows, fim = [], None
    for _ in range(paginas):
        p = {"category": "spot", "symbol": simbolo,
             "interval": mp[intervalo], "limit": 1000}
        if fim:
            p["end"] = fim
        k = _get("https://api.bybit.com/v5/market/kline", p)["result"]["list"]
        if not k:
            break
        rows += k
        fim = int(k[-1][0]) - 1
        if len(k) < 1000:
            break
    return _ohlcv([r[:6] for r in rows], 1, (0, 1, 2, 3, 4, 5))


def de_kucoin(simbolo, intervalo, paginas):
    mp = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
          "1h": "1hour", "4h": "4hour", "1d": "1day"}
    par = simbolo.replace("USDT", "-USDT")
    rows, fim = [], None
    for _ in range(paginas):
        p = {"symbol": par, "type": mp[intervalo]}
        if fim:
            p["endAt"] = fim
        k = _get("https://api.kucoin.com/api/v1/market/candles", p)["data"]
        if not k:
            break
        rows += k
        fim = int(k[-1][0]) - 1
        time.sleep(0.2)
    # KuCoin: [t(s), open, close, high, low, vol]
    return _ohlcv([r[:6] for r in rows], 1000, (0, 1, 3, 4, 2, 5))


def de_okx(simbolo, intervalo, paginas):
    mp = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
          "1h": "1H", "4h": "4H", "1d": "1D"}
    par = simbolo.replace("USDT", "-USDT")
    rows, fim = [], None
    for _ in range(paginas * 10):
        p = {"instId": par, "bar": mp[intervalo], "limit": 100}
        if fim:
            p["after"] = fim
        k = _get("https://www.okx.com/api/v5/market/history-candles", p)["data"]
        if not k:
            break
        rows += [r for r in k if r[8] == "1"]
        fim = k[-1][0]
        time.sleep(0.1)
    return _ohlcv([r[:6] for r in rows], 1, (0, 1, 2, 3, 4, 5))


def de_yahoo(simbolo, intervalo, paginas):
    import yfinance as yf
    base = simbolo.replace("USDT", "")
    ult = None
    for t in (f"{base}-USD", f"{base}21794-USD"):  # APT no Yahoo: APT21794-USD
        kw = ({"period": "730d", "interval": "1h"} if intervalo == "1h"
              else {"period": "5y", "interval": "1d"})
        df = yf.download(t, auto_adjust=True, progress=False, **kw)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna()
        if len(df) > 500:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Volume"]]
        ult = df
    return ult


FONTES = [("Binance", de_binance), ("Binance Vision", de_binance_vision),
          ("Bybit", de_bybit), ("KuCoin", de_kucoin), ("OKX", de_okx),
          ("Yahoo", de_yahoo)]


def sintetico():
    rng = np.random.default_rng(0)
    n = 6000
    p = 30000 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({"Open": p, "High": p * 1.003, "Low": p * 0.997,
                         "Close": p, "Volume": rng.integers(100, 900, n)},
                        index=idx)


_PREF = None


def carregar(simbolo, intervalo, csv, paginas=20, permitir_sintetico=False,
             silencioso=False):
    if csv:
        df = pd.read_csv(csv, parse_dates=[0], index_col=0)
        df.columns = [c.strip().capitalize() for c in df.columns]
        return df.sort_index().dropna(), "CSV"
    if permitir_sintetico:
        return sintetico(), "SINTÉTICO (teste)"
    global _PREF
    for nome, fn in sorted(FONTES, key=lambda f: f[0] != _PREF):
        try:
            if not silencioso:
                print(f"Tentando {nome}...", flush=True)
            df = fn(simbolo, intervalo, paginas)
            if df is not None and len(df) > 1000:
                _PREF = nome
                return df.iloc[:-1], nome   # descarta o candle ainda aberto
            if silencioso and df is not None and len(df) > 0:
                raise SystemExit("poucos dados")
            if not silencioso:
                print(f"  {nome}: poucos dados.")
        except SystemExit:
            raise
        except Exception as e:
            if not silencioso:
                print(f"  {nome} falhou: {type(e).__name__} {str(e)[:80]}")
    raise SystemExit("Nenhuma fonte de dados respondeu. Use --csv.")

# -------------------------------------------------------------- features
def rsi(c, n=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()      # RSI de Wilder
    p = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + g / (p + 1e-9))


def criar_features(df):
    """Retorna features; ret_fut/alvo ficam NaN nos últimos candles (sem futuro)."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = c.pct_change(n)
    f["rsi"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - ma20) / (2 * sd20 + 1e-9)
    f["dist_ma50"] = c / c.rolling(50).mean() - 1
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    f["vol_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std() + 1e-9)
    f["volat_24"] = c.pct_change().rolling(24).std()
    if "TakerRatio" in df:  # pressão compradora (só Binance)
        f["taker"] = df["TakerRatio"].rolling(6).mean()
    f["hora_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    f["hora_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    f["dia_semana"] = df.index.dayofweek
    f = f.dropna().copy()
    f["ret_fut"] = (c.shift(-HORIZONTE) / c - 1).reindex(f.index)
    f["alvo"] = (f["ret_fut"] > LIMIAR).astype(float)
    f.loc[f["ret_fut"].isna(), "alvo"] = np.nan
    return f


def novo_modelo():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                          max_iter=200, l2_regularization=1.0)


def sem_sobreposicao(pos):
    """Escolhe entradas que não se sobrepõem (uma por janela de HORIZONTE)."""
    ok, ult = [], -10**9
    for p in pos:
        if p >= ult + HORIZONTE:
            ok.append(p)
            ult = p
    return np.array(ok, dtype=int)

# =========================================================== APP (scalp v4)
import re
import matplotlib
import matplotlib.pyplot as plt

try:
    from IPython.display import display, HTML
    def mostrar(h): display(HTML(h))
except ImportError:
    def mostrar(h): print(re.sub(r"<[^>]+>", " ", h))

FUSO = "America/Sao_Paulo"      # fuso de TODOS os horários mostrados
N_SAIDA = 12                    # duração máx. da operação (velas); ajustado por _config()
ATR_STOP, ATR_ALVO = 1.5, 2.5   # stop/alvo em ATR (relação ganho/risco ≈ 1,7:1)
MIN_ALVO = 0.01                 # alvo mínimo de 1% (= 5x o custo de ida e volta)
PONT_MAX = 22
MINUTOS = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def _config(intervalo):
    """Duração da operação ≈ 3h, em velas do intervalo escolhido."""
    global N_SAIDA, HORIZONTE
    N_SAIDA = int(np.clip(round(180 / MINUTOS.get(intervalo, 15)), 6, 36))
    HORIZONTE = N_SAIDA


def local(ts):
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert(FUSO)


def normalizar(token):
    t = re.sub(r"[^A-Z0-9]", "", token.upper())
    return t if t.endswith("USDT") else t + "USDT"


def _wilder(x, n=14):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def smc(d, k_imp=1.5, vida=60):
    """Zonas de smart money (sem olhar o futuro): Order Blocks e Fair Value Gaps."""
    o, h, l, c = (d[x].values for x in ("o", "h", "l", "c"))
    atr, rv = d["atr"].values, d["rvol"].values
    n = len(d)
    Z = {k: np.full(n, np.nan) for k in ("obb_lo", "obb_hi", "obs_lo", "obs_hi",
                                         "fvb_lo", "fvb_hi", "fvs_lo", "fvs_hi")}
    obb = obs = fvb = fvs = None            # (lo, hi, vela_de_criação)
    for t in range(n):
        if obb and (c[t] < obb[0] or t - obb[2] > vida): obb = None   # quebrada ou velha
        if obs and (c[t] > obs[1] or t - obs[2] > vida): obs = None
        if fvb and (c[t] < fvb[0] or t - fvb[2] > vida): fvb = None
        if fvs and (c[t] > fvs[1] or t - fvs[2] > vida): fvs = None
        if t >= 2:
            if l[t] > h[t - 2] and c[t - 1] > o[t - 1]: fvb = (h[t - 2], l[t], t)
            if h[t] < l[t - 2] and c[t - 1] < o[t - 1]: fvs = (h[t], l[t - 2], t)
        if np.isfinite(atr[t]) and np.isfinite(rv[t]) and rv[t] >= 1.2:
            corpo = c[t] - o[t]
            if corpo >= k_imp * atr[t]:                     # impulso de alta -> OB = última vela vendedora
                for j in range(t - 1, max(t - 6, -1), -1):
                    if c[j] < o[j]:
                        obb = (l[j], h[j], t); break
            elif -corpo >= k_imp * atr[t]:                  # impulso de baixa -> OB = última vela compradora
                for j in range(t - 1, max(t - 6, -1), -1):
                    if c[j] > o[j]:
                        obs = (l[j], h[j], t); break
        for z, nm in ((obb, "obb"), (obs, "obs"), (fvb, "fvb"), (fvs, "fvs")):
            if z and z[2] < t:                              # não vale na própria vela de criação
                Z[nm + "_lo"][t], Z[nm + "_hi"][t] = z[0], z[1]
    for k, v in Z.items():
        d[k] = v
    return d


def indicadores(df, k=5):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    d = pd.DataFrame({"o": df["Open"], "h": h, "l": l, "c": c, "v": v})
    d["ma20"], d["ma50"], d["ma200"] = (c.rolling(20).mean(), c.rolling(50).mean(),
                                         c.rolling(200).mean())
    sd = c.rolling(20).std(ddof=0)                           # padrão Bollinger
    d["bb_up"], d["bb_lo"] = d["ma20"] + 2 * sd, d["ma20"] - 2 * sd
    d["bb_w"] = (d["bb_up"] - d["bb_lo"]) / d["ma20"]
    d["rsi"] = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["mh"] = macd - macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    d["atr"] = _wilder(tr, 14)                               # ATR de Wilder
    d["rvol"] = v / (v.rolling(24).mean().shift() + 1e-9)    # média das 24 velas ANTERIORES
    d["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    d["obv_ma"] = d["obv"].ewm(span=24, adjust=False).mean()
    d["max24"] = h.rolling(24).max().shift()
    d["sq"] = d["bb_w"].rolling(200, min_periods=100).rank(pct=True) <= 0.15
    tem_taker = "TakerRatio" in df
    d["taker"] = df["TakerRatio"].rolling(6).mean() if tem_taker else np.nan
    d["cvd"] = (v * (2 * df["TakerRatio"] - 1)).cumsum() if tem_taker else d["obv"]
    # --- estrutura: pivôs confirmados k velas depois (sem olhar o futuro)
    ar = pd.Series(np.arange(len(d), dtype=float), index=d.index)
    ish = h == h.rolling(2 * k + 1, center=True).max()
    isl = l == l.rolling(2 * k + 1, center=True).min()
    d["sh"], d["sl"] = h.where(ish).shift(k).ffill(), l.where(isl).shift(k).ffill()
    sh_i, sl_i = ar.where(ish).shift(k).ffill(), ar.where(isl).shift(k).ffill()
    d["up_leg"] = sl_i < sh_i
    d["leg"] = d["sh"] - d["sl"]
    d["leg_ok"] = d["leg"] >= 3 * d["atr"]                   # perna relevante (>= 3 ATR)
    d["retr"] = np.where(d["up_leg"], (d["sh"] - d["c"]) / d["leg"],
                         (d["c"] - d["sl"]) / d["leg"])
    # --- padrões de vela, só com contexto (reversão após 3 velas na direção oposta)
    body = (d["c"] - d["o"]).abs()
    rng = (d["h"] - d["l"]) + 1e-12
    lw = np.minimum(d["o"], d["c"]) - d["l"]
    uw = d["h"] - np.maximum(d["o"], d["c"])
    po, pc = d["o"].shift(), d["c"].shift()
    eng_b = (d["c"] > d["o"]) & (pc < po) & (d["c"] >= po) & (d["o"] <= pc)
    eng_s = (d["c"] < d["o"]) & (pc > po) & (d["c"] <= po) & (d["o"] >= pc)
    pin_b = (lw >= 2 * body) & (lw >= .5 * rng) & (uw <= .3 * rng)
    pin_s = (uw >= 2 * body) & (uw >= .5 * rng) & (lw <= .3 * rng)
    d["cand_b"] = (pin_b | eng_b) & (d["c"] < d["c"].shift(3))
    d["cand_s"] = (pin_s | eng_s) & (d["c"] > d["c"].shift(3))
    # --- varredura de liquidez (caça de stops): rompe a máx/mín de 20 velas e volta
    ll20, hh20 = l.rolling(20).min().shift(), h.rolling(20).max().shift()
    sw_b = (l < ll20) & (c > ll20) & (lw >= .4 * rng)
    sw_s = (h > hh20) & (c < hh20) & (uw >= .4 * rng)
    d["sw_b3"] = sw_b.astype(int).rolling(3).max().fillna(0).astype(bool)
    d["sw_s3"] = sw_s.astype(int).rolling(3).max().fillna(0).astype(bool)
    return smc(d)


def pontuar(d, lado="long"):
    """Pontuação -22..+22 a favor do lado (sem olhar o futuro)."""
    I = lambda x: x.astype(int)
    up = d["c"] > d["o"]
    alto = d["rvol"] > 1.8
    bull = (d["c"] > d["ma50"]) & (d["ma20"] > d["ma50"])
    bear = (d["c"] < d["ma50"]) & (d["ma20"] < d["ma50"])
    mup = (d["mh"] > 0) & (d["mh"] > d["mh"].shift())
    mdn = (d["mh"] < 0) & (d["mh"] < d["mh"].shift())
    min24 = d["l"].rolling(24).min().shift()
    zona = (d["retr"] >= .35) & (d["retr"] <= .65) & d["leg_ok"]
    nlow = (d["l"] - d["sl"]).abs() <= .5 * d["atr"]
    nhigh = (d["h"] - d["sh"]).abs() <= .5 * d["atr"]
    htf_up = (d["c"] > d["ma200"]) & (d["ma50"] > d["ma50"].shift(20))
    htf_dn = (d["c"] < d["ma200"]) & (d["ma50"] < d["ma50"].shift(20))
    dist = np.maximum(ATR_ALVO * d["atr"], MIN_ALVO * d["c"]) * .8
    fl, p24 = d["cvd"], d["c"].shift(24)
    acc = (d["c"] < p24) & (fl > fl.shift(24))        # preço cai, fluxo comprador sobe
    dis = (d["c"] > p24) & (fl < fl.shift(24))        # preço sobe, fluxo vendedor sobe
    ob_b = d["obb_lo"].notna() & (d["l"] <= d["obb_hi"]) & (d["c"] >= d["obb_lo"]) & up
    ob_s = d["obs_lo"].notna() & (d["h"] >= d["obs_lo"]) & (d["c"] <= d["obs_hi"]) & ~up
    fv_b = d["fvb_lo"].notna() & (d["l"] <= d["fvb_hi"]) & (d["c"] >= d["fvb_lo"]) & up
    fv_s = d["fvs_lo"].notna() & (d["h"] >= d["fvs_lo"]) & (d["c"] <= d["fvs_hi"]) & ~up
    if lado == "long":
        s = 2 * I(bull) - 2 * I(bear) + I(htf_up) - I(htf_dn)
        s += I(mup) - I(mdn)
        s += I((d["rsi"] >= 45) & (d["rsi"] <= 68)) - I((d["rsi"] > 75) | (d["rsi"] < 35))
        s += 2 * I(alto & up) - 2 * I(alto & ~up)
        s += I(d["obv"] > d["obv_ma"]) - I(d["obv"] <= d["obv_ma"])
        s += 2 * I((d["c"] > d["bb_up"]) & (d["rvol"] > 1.5)) + I(d["c"] > d["max24"])
        s += I(d["taker"] > 0.55) - I(d["taker"] < 0.45)
        s += 2 * I(zona & d["up_leg"] & up) + I(d["cand_b"]) - I(d["cand_s"])
        s += I(nlow & (d["c"] > d["sl"]) & up)
        s += 2 * I(d["sw_b3"] & up) + 2 * I(ob_b) + I(fv_b) + I(acc) - I(dis)    # smart money
        s -= I((d["sh"] > d["c"]) & ((d["sh"] - d["c"]) < dist))                 # resistência no caminho
    else:
        s = 2 * I(bear) - 2 * I(bull) + I(htf_dn) - I(htf_up)
        s += I(mdn) - I(mup)
        s += I((d["rsi"] >= 32) & (d["rsi"] <= 55)) - I((d["rsi"] < 25) | (d["rsi"] > 65))
        s += 2 * I(alto & ~up) - 2 * I(alto & up)
        s += I(d["obv"] < d["obv_ma"]) - I(d["obv"] >= d["obv_ma"])
        s += 2 * I((d["c"] < d["bb_lo"]) & (d["rvol"] > 1.5)) + I(d["c"] < min24)
        s += I(d["taker"] < 0.45) - I(d["taker"] > 0.55)
        s += 2 * I(zona & ~d["up_leg"] & ~up) + I(d["cand_s"]) - I(d["cand_b"])
        s += I(nhigh & (d["c"] < d["sh"]) & ~up)
        s += 2 * I(d["sw_s3"] & ~up) + 2 * I(ob_s) + I(fv_s) + I(dis) - I(acc)
        s -= I((d["sl"] < d["c"]) & ((d["c"] - d["sl"]) < dist))                 # suporte no caminho
    return s


def niveis(e, atr, lado):
    """Alvo >= 1% (cobre os custos); stop proporcional (R:R = ATR_ALVO/ATR_STOP)."""
    ap = max(ATR_ALVO * atr / e, MIN_ALVO)
    sp = ap * ATR_STOP / ATR_ALVO
    if lado == "long":
        return {"entrada": e, "stop": e * (1 - sp), "alvo": e * (1 + ap), "sp": sp, "ap": ap}
    return {"entrada": e, "stop": e * (1 + sp), "alvo": e * (1 - ap), "sp": sp, "ap": ap}


def simular(d, i, lado="long"):
    e = d["c"].iloc[i]
    n = niveis(e, d["atr"].iloc[i], lado)
    sg = 1 if lado == "long" else -1
    for j in range(i + 1, min(i + 1 + N_SAIDA, len(d))):
        h, l = d["h"].iloc[j], d["l"].iloc[j]
        if (l <= n["stop"]) if lado == "long" else (h >= n["stop"]):   # stop primeiro
            return sg * (n["stop"] / e - 1) - 2 * CUSTO
        if (h >= n["alvo"]) if lado == "long" else (l <= n["alvo"]):
            return sg * (n["alvo"] / e - 1) - 2 * CUSTO
    j = min(i + N_SAIDA, len(d) - 1)
    return sg * (d["c"].iloc[j] / e - 1) - 2 * CUSTO


def backtest(d, mask, lado):
    idx = np.flatnonzero(mask.values & d["atr"].notna().values)
    idx = idx[idx < len(d) - 1 - N_SAIDA]
    pos, ult = [], -10**9
    for i in idx:
        if i >= ult + N_SAIDA:
            pos.append(i)
            ult = i
    pos = np.array(pos, dtype=int)
    r = np.array([simular(d, i, lado) for i in pos])
    base = np.array([simular(d, i, lado) for i in range(60, len(d) - 1 - N_SAIDA, N_SAIDA)])
    return pos, r, base


def modelo_dir(bruto):
    """ML: probabilidade de o preço estar mais alto em HORIZONTE velas."""
    f = criar_features(bruto)
    rot = f[f["ret_fut"].notna()]
    X, y = rot.drop(columns=["alvo", "ret_fut"]), (rot["ret_fut"] > 0).astype(int)
    prob = pd.Series(np.nan, index=X.index)
    for tr, te in TimeSeriesSplit(n_splits=5, gap=HORIZONTE).split(X):
        prob.iloc[te] = novo_modelo().fit(X.iloc[tr], y.iloc[tr]).predict_proba(
            X.iloc[te])[:, 1]
    ok = prob.notna()
    auc = roc_auc_score(y[ok], prob[ok])
    fin = novo_modelo().fit(X.iloc[:-HORIZONTE], y.iloc[:-HORIZONTE])
    xa = f.drop(columns=["alvo", "ret_fut"]).iloc[[-1]]
    return fin.predict_proba(xa)[0, 1], auc, y.mean()


# ---------------------------------------------------- contexto BTC / ETH
def contexto(intervalo, paginas=6):
    """Tendência de BTC e ETH (mesmo critério técnico) para ponderar qualquer token."""
    ctx = {}
    for nome, sym in (("btc", "BTCUSDT"), ("eth", "ETHUSDT")):
        try:
            bruto, _ = carregar(sym, intervalo, None, paginas, False, True)
            d = indicadores(bruto)
            bias = ((pontuar(d, "long") - pontuar(d, "short")) / (2 * PONT_MAX)).clip(-1, 1)
            ctx[nome] = dict(d=d, bias=bias.rolling(3, min_periods=1).mean())
        except (Exception, SystemExit):      # sem dados de BTC/ETH: segue sem o filtro de mercado
            ctx[nome] = None
    return ctx


def mercado(d, ctx, simbolo, janela=500):
    """Série 'mk' (-1..+1): viés de BTC/ETH ponderado pela correlação do token com cada um."""
    mk, info = pd.Series(0.0, index=d.index), {}
    for nome, w in (("btc", .6), ("eth", .4)):
        x = (ctx or {}).get(nome)
        if x is None or simbolo.startswith(nome.upper()):      # não conta o token com ele mesmo
            continue
        b = x["bias"].reindex(d.index, method="ffill").fillna(0)
        r2 = x["d"]["c"].pct_change().reindex(d.index)
        cor = d["c"].pct_change().iloc[-janela:].corr(r2.iloc[-janela:])
        cor = 0.0 if not np.isfinite(cor) else max(cor, 0.0)
        mk = mk + w * cor * b
        info[nome] = dict(bias=float(b.iloc[-1]), cor=float(cor))
    return mk, info


# ------------------------------------------------------------- decisão
def direcao(d, ml=None, mk=None, info=None):
    """Sempre decide LONG ou SHORT: técnica + mercado (BTC/ETH) + (opcional) ML."""
    mk = pd.Series(0.0, index=d.index) if mk is None else mk
    sL, sS = pontuar(d, "long"), pontuar(d, "short")
    ds = sL - sS
    zs = ds / (2 * PONT_MAX)                       # leitura técnica em cada vela
    z = zs.iloc[-1]
    zml = 0.0 if ml is None else float(np.clip((ml[0] - ml[2]) / 0.15, -1, 1))
    cs = zs + 0.35 * mk                            # técnica + mercado (BTC/ETH), sem ML
    cn = cs.iloc[-1]
    comb = cn + 0.2 * zml
    sobe = d["ma20"].iloc[-1] > d["ma50"].iloc[-1]
    lado = "long" if (comb > 0 or (comb == 0 and sobe)) else "short"
    sg = 1 if lado == "long" else -1
    thr = max(0.12, 0.6 * abs(cn))                 # histórico: mesma força combinada, mesma direção
    mask = (cs >= thr) if lado == "long" else (cs <= -thr)
    pos, r, rb = backtest(d, mask, lado)
    a_fav = (mk.values[pos] * sg >= 0) if len(pos) else np.array([], bool)   # BTC/ETH não contra
    u = d.iloc[-1]
    nv = niveis(u["c"], u["atr"], lado)
    be = (nv["sp"] + 2 * CUSTO) / (nv["sp"] + nv["ap"])
    viab = min(1.0, ATR_ALVO * u["atr"] / u["c"] / MIN_ALVO)    # alvo mínimo é alcançável?
    return dict(lado=lado, z=z, mk=float(mk.iloc[-1]), sg=sg, sL=sL.iloc[-1], sS=sS.iloc[-1],
                comb=comb, ml=ml, pos=pos, r=r, r_a=r[a_fav], r_c=r[~a_fav] if len(r) else r,
                rb=rb, be=be, sp=nv["sp"], ap=nv["ap"], viab=viab, mk_ser=mk, info=info or {})


def confianca(i):
    r = i["r_a"] if len(i["r_a"]) >= 10 else i["r"]
    ok = len(r) >= 15 and r.mean() > 0 and (r > 0).mean() > i["be"]
    f = i["z"] * i["sg"]                       # apoio técnico NA direção final
    nivel = 2 if (ok and f >= .30) else 1 if (ok or f >= .25) else 0
    if i["mk"] * i["sg"] < -0.15:
        nivel -= 1                                  # BTC/ETH contra a direção
    if i["viab"] < 0.7:
        nivel -= 1                                  # token pouco volátil para o alvo mínimo
    return ["fraca", "média", "forte"][max(nivel, 0)]


def chave(i):
    r = i["r_a"] if len(i["r_a"]) >= 10 else i["r"]
    k = abs(i["comb"]) - .3 * (1 - i["viab"])      # força combinada (técnica + mercado + ML)
    if len(r) >= 10:
        k += .2 * (r.mean() > 0) + .1 * ((r > 0).mean() > i["be"])
    return k


def razoes(d, lado):
    u, a = d.iloc[-1], []
    tend = ("alta" if u["c"] > u["ma50"] and u["ma20"] > u["ma50"] else
            "baixa" if u["c"] < u["ma50"] and u["ma20"] < u["ma50"] else "lateral")
    a.append(f"Tendência de {tend} (preço vs. MA20/MA50) · "
             f"{'acima' if u['c'] > u['ma200'] else 'abaixo'} da MA200 · MACD "
             f"{'positivo' if u['mh'] > 0 else 'negativo'} · RSI {u['rsi']:.0f}.")
    if u["leg_ok"] and np.isfinite(u["retr"]):
        zona = .35 <= u["retr"] <= .65
        a.append(f"Fibonacci: última perna de {'alta' if u['up_leg'] else 'baixa'}; preço em "
                 f"{u['retr']:.0%} do retrocesso ("
                 f"{'ZONA DE OURO 38–62%: região típica de retomada' if zona else 'fora da zona de ouro'}).")
    if u["cand_b"]:
        a.append("Padrão de vela altista após queda (martelo/engolfo de alta).")
    if u["cand_s"]:
        a.append("Padrão de vela baixista após alta (estrela cadente/engolfo de baixa).")
    if abs(u["l"] - u["sl"]) <= .5 * u["atr"]:
        a.append(f"Preço testando suporte em {u['sl']:.6g}.")
    if abs(u["h"] - u["sh"]) <= .5 * u["atr"]:
        a.append(f"Preço testando resistência em {u['sh']:.6g}.")
    return a + alertas(d)


def smc_texto(d, lado):
    """Leitura de 'smart money' a partir de candles e volume (heurística, não dado de carteira)."""
    u, a = d.iloc[-1], []
    tag = lambda dir_: "✅ a favor" if dir_ == lado else "⚠️ contra"
    if d["sw_b3"].iloc[-1]:
        a.append(f"Varredura de liquidez abaixo da mínima recente com retorno (possível caça de stops "
                 f"e compra institucional) — {tag('long')}.")
    if d["sw_s3"].iloc[-1]:
        a.append(f"Varredura de liquidez acima da máxima recente com retorno (possível caça de stops "
                 f"e venda institucional) — {tag('short')}.")
    if np.isfinite(u["obb_lo"]) and u["l"] <= u["obb_hi"] and u["c"] >= u["obb_lo"]:
        a.append(f"Reteste de Order Block altista ({u['obb_lo']:.6g}–{u['obb_hi']:.6g}) — {tag('long')}.")
    if np.isfinite(u["obs_lo"]) and u["h"] >= u["obs_lo"] and u["c"] <= u["obs_hi"]:
        a.append(f"Reteste de Order Block baixista ({u['obs_lo']:.6g}–{u['obs_hi']:.6g}) — {tag('short')}.")
    if np.isfinite(u["fvb_lo"]) and u["l"] <= u["fvb_hi"] and u["c"] >= u["fvb_lo"]:
        a.append(f"Preço preenchendo Fair Value Gap altista — {tag('long')}.")
    if np.isfinite(u["fvs_lo"]) and u["h"] >= u["fvs_lo"] and u["c"] <= u["fvs_hi"]:
        a.append(f"Preço preenchendo Fair Value Gap baixista — {tag('short')}.")
    if len(d) > 25:
        up24, fl = u["c"] > d["c"].iloc[-25], u["cvd"] > d["cvd"].iloc[-25]
        if (not up24) and fl:
            a.append(f"Acumulação: preço caiu, mas o fluxo comprador (delta de volume) subiu — {tag('long')}.")
        if up24 and (not fl):
            a.append(f"Distribuição: preço subiu, mas o fluxo vendedor predominou — {tag('short')}.")
    return a or ["Sem sinal claro de smart money nas últimas velas."]


def alertas(d):
    u, a = d.iloc[-1], []
    if u["sq"]:
        a.append("Compressão de volatilidade (faixa de Bollinger entre as 15% mais "
                 "estreitas do histórico): costuma anteceder movimento forte; a "
                 "direção só se confirma no rompimento com volume.")
    if u["rvol"] > 1.5 and abs(u["c"] - u["o"]) < 0.5 * u["atr"]:
        a.append("Volume alto sem deslocamento do preço (absorção): pode anteceder rompimento.")
    if len(d) > 25:
        p0, o0 = d["c"].iloc[-25], d["obv"].iloc[-25]
        if u["c"] < p0 and u["obv"] > o0:
            a.append("Divergência altista: preço caiu nas últimas 24 velas, mas o OBV subiu (acumulação).")
        if u["c"] > p0 and u["obv"] < o0:
            a.append("Divergência baixista: preço subiu, mas o OBV caiu (alta sem volume real).")
    if u["c"] > u["max24"]:
        a.append("Rompeu a máxima das últimas 24 velas.")
    if u["rsi"] > 75:
        a.append("RSI > 75: esticado, risco de correção.")
    if u["rsi"] < 30:
        a.append("RSI < 30: sobrevendido (pode quicar, mas sem confirmação não é entrada).")
    if u["rvol"] > 1.8:
        a.append(f"Volume {u['rvol']:.1f}x acima da média das últimas 24 velas.")
    return a or ["Nenhum alerta de antecipação no momento."]


def grafico(d, i, plano, titulo, n=160):
    x, t = d.iloc[-n:], np.arange(min(n, len(d)))
    BG, FG, UP, DN = "#0e1117", "#d0d4dc", "#26a69a", "#ef5350"
    fig, ax = plt.subplots(4, 1, figsize=(9, 12), sharex=True,
                           gridspec_kw={"height_ratios": [5, 1.6, 1.4, 1.4]})
    fig.patch.set_facecolor(BG)
    for a in ax:
        a.set_facecolor(BG)
        a.tick_params(colors=FG, labelsize=8)
        a.grid(alpha=.12)
        for s in a.spines.values():
            s.set_color("#333")
    cor = np.where(x["c"] >= x["o"], UP, DN)
    a = ax[0]
    a.vlines(t, x["l"], x["h"], color=cor, lw=.8)
    a.bar(t, (x["c"] - x["o"]).abs(), bottom=np.minimum(x["c"], x["o"]), color=cor, width=.7)
    a.plot(t, x["ma20"], color="#ffb300", lw=1, label="MA20")
    a.plot(t, x["ma50"], color="#42a5f5", lw=1, label="MA50")
    a.fill_between(t, x["bb_lo"], x["bb_up"], color="#9e9e9e", alpha=.12, label="Bollinger")
    lo = min(x["l"].min(), plano["stop"], plano["alvo"])
    hi = max(x["h"].max(), plano["stop"], plano["alvo"])
    a.set_ylim(lo * .995, hi * 1.005)
    ylo, yhi = a.get_ylim()
    u = d.iloc[-1]
    if u["leg_ok"] and np.isfinite(u["leg"]):                       # Fibonacci
        for r_, cc in ((0, "#888"), (.382, "#ffd54f"), (.5, "#ffb74d"),
                       (.618, "#ff8a65"), (.786, "#bdbdbd"), (1, "#888")):
            lv = u["sh"] - r_ * u["leg"] if u["up_leg"] else u["sl"] + r_ * u["leg"]
            if ylo <= lv <= yhi:
                a.axhline(lv, color=cc, lw=.7, ls=":")
                a.text(len(x) - 1, lv, f"Fib {r_:.3f} ", color=cc, fontsize=7, ha="right", va="bottom")
    long_ = i["lado"] == "long"
    zonas = (("obb", "#00e676", "Order Block"), ("fvb", "#69f0ae", "FVG")) if long_ else \
            (("obs", "#ff5252", "Order Block"), ("fvs", "#ff8a80", "FVG"))
    for pre, cc, nm in zonas:
        if np.isfinite(u[pre + "_lo"]):
            a.axhspan(u[pre + "_lo"], u[pre + "_hi"], color=cc, alpha=.13, label=f"{nm} {'alta' if long_ else 'baixa'}")
    sw = x["sw_b3"] if long_ else x["sw_s3"]
    ps_sw = [k for k in range(len(x)) if sw.iloc[k]]
    if ps_sw:
        yy = x["l"].iloc[ps_sw] * .998 if long_ else x["h"].iloc[ps_sw] * 1.002
        a.scatter(ps_sw, yy, marker="*", color="#ffeb3b", s=60, zorder=6, label="varredura de liquidez")
    off = len(d) - len(x)
    ps = [p - off for p in i["pos"] if p >= off]
    if ps:
        if long_:
            a.scatter(ps, x["l"].iloc[ps] * .999, marker="^", color="#00e676", s=40, zorder=5, label="sinais passados")
        else:
            a.scatter(ps, x["h"].iloc[ps] * 1.001, marker="v", color="#ff1744", s=40, zorder=5, label="sinais passados")
    for k, cc in (("entrada", "#ffffff"), ("stop", DN), ("alvo", UP)):
        a.axhline(plano[k], color=cc, ls="--", lw=1.3)
        a.text(0, plano[k], f" {k.upper()} {plano[k]:.6g}", color=cc, fontsize=9, va="bottom", fontweight="bold")
    a.set_title(titulo, color=FG, fontsize=12, fontweight="bold")
    a.legend(loc="lower left", fontsize=7, facecolor=BG, labelcolor=FG, ncol=4)
    ax[1].bar(t, x["v"], color=cor, width=.8)
    ax[1].plot(t, x["v"].rolling(24).mean(), color="#ffb300", lw=1)
    ax[1].set_ylabel("Volume", color=FG, fontsize=8)
    ax[2].plot(t, x["rsi"], color="#ab47bc", lw=1)
    for lv in (30, 50, 70):
        ax[2].axhline(lv, color="#555", lw=.7, ls=":")
    ax[2].set_ylabel("RSI", color=FG, fontsize=8)
    ax[3].bar(t, x["mh"], color=np.where(x["mh"] >= 0, UP, DN), width=.8)
    ax[3].set_ylabel("MACD", color=FG, fontsize=8)
    ax[3].set_xticks(t[::20])
    ax[3].set_xticklabels([f"{local(ts):%d/%m %H:%M}" for ts in x.index[::20]], rotation=45)
    ax[3].set_xlabel(f"horário: {FUSO}", color=FG, fontsize=8)
    fig.tight_layout()
    return fig


def _caixa(rot, val, sub=""):
    return ("<div style='flex:1;min-width:95px;background:rgba(0,0,0,.28);border-radius:10px;"
            "padding:8px;text-align:center'>"
            f"<div style='font-size:12px;opacity:.85'>{rot}</div>"
            f"<div style='font-size:20px;font-weight:700'>{val}</div>"
            f"<div style='font-size:12px;opacity:.85'>{sub}</div></div>")


def preco_agora(sym):
    for b in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            return float(_get(b + "/api/v3/ticker/price", {"symbol": sym})["price"])
        except Exception:
            pass
    return None


def _seta(b):
    return "⬆️ alta" if b > .15 else "⬇️ queda" if b < -.15 else "➡️ lateral"


def mercado_texto(i):
    if not i["info"]:
        return ""
    partes = " · ".join(f"{k.upper()} {_seta(v['bias'])} ({v['bias']:+.2f}, correl. {v['cor']:.2f})"
                        for k, v in i["info"].items())
    adj = i["mk"] * i["sg"]
    veredito = ("✅ a favor da direção" if adj > .1 else "⚠️ CONTRA a direção (risco maior)" if adj < -.1
                else "neutro")
    return f"<div style='margin-top:6px;font-size:13px'>📈 Mercado: {partes} → {veredito}.</div>"


def apresentar(simbolo, intervalo, fonte, bruto, d, i, png=None, posicao=None):
    u, lado = d.iloc[-1], i["lado"]
    plano = niveis(u["c"], u["atr"], lado)
    e = plano["entrada"]
    m = MINUTOS.get(intervalo, 60)
    abre = local(d.index[-1])
    fecha = abre + pd.Timedelta(minutes=m)
    agora = pd.Timestamp.now(tz="UTC").tz_convert(FUSO)
    idade = (agora - fecha).total_seconds() / 60
    sair = fecha + pd.Timedelta(minutes=m * N_SAIDA)
    conf = confianca(i)
    long_ = lado == "long"
    cor = "#1b8a5a" if long_ else "#b3343a"
    titulo = "⬆️ LONG (COMPRA)" if long_ else "⬇️ SHORT (VENDA)"
    pa = preco_agora(simbolo)
    aviso = ""
    if pa and abs(pa - e) > 0.3 * plano["sp"] * e:
        aviso += (f"<div style='margin-top:6px;font-size:13px'>⚠️ O preço agora ({pa:.6g}) já se afastou da "
                  f"entrada calculada ({e:.6g}); reavalie antes de entrar.</div>")
    if idade > 2 * m:
        aviso += (f"<div style='margin-top:6px;font-size:13px'>⚠️ Dados com {idade:.0f} min de atraso: "
                  f"o cenário pode ter mudado.</div>")
    if i["viab"] < .7:
        aviso += ("<div style='margin-top:6px;font-size:13px'>⚠️ Token pouco volátil neste intervalo: o alvo "
                  f"mínimo de {MIN_ALVO:.0%} é maior que o movimento típico; a chance de atingi-lo é menor.</div>")
    custo = 2 * CUSTO
    g, p = 1000 * (plano["ap"] - custo), 1000 * (plano["sp"] + custo)
    alav = " · ".join(f"{L}x: {plano['sp'] * L:.0%}" for L in (5, 10, 20))
    r = i["r_a"] if len(i["r_a"]) >= 10 else i["r"]
    if len(i["r"]):
        hist = (f"Sinais parecidos neste token no histórico: <b>{(i['r'] > 0).mean():.0%}</b> de acerto em "
                f"{len(i['r'])} operações (mínimo p/ empatar: {i['be']:.0%}), retorno médio líquido {i['r'].mean():.2%}.")
        if len(i["r_a"]) >= 5 and len(i["r_c"]) >= 5:
            hist += (f" Com BTC/ETH <b>a favor</b>: {(i['r_a'] > 0).mean():.0%} (n={len(i['r_a'])}) · "
                     f"<b>contra</b>: {(i['r_c'] > 0).mean():.0%} (n={len(i['r_c'])}).")
    else:
        hist = "Sem sinais parecidos suficientes no histórico deste token."
    ml = (f" Modelo ML: {i['ml'][0]:.0%} de chance de o preço estar acima em {HORIZONTE} velas "
          f"(média histórica {i['ml'][2]:.0%}).") if i["ml"] else ""
    pos = f"#{posicao} no ranking · " if posicao else ""
    mostrar(
        f"<div style='background:{cor};color:#fff;padding:16px;border-radius:14px;font-family:sans-serif;font-size:15px'>"
        f"<div style='font-size:13px;opacity:.9'>{pos}{simbolo} · {intervalo} · {fonte}</div>"
        f"<div style='font-size:30px;font-weight:800'>{titulo}</div>"
        f"<div style='font-size:14px'>Confiança: <b>{conf}</b> · técnica long {i['sL']:+.0f} × short {i['sS']:+.0f} (máx. {PONT_MAX})</div>"
        "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:10px'>"
        + _caixa("ENTRADA", f"{e:.6g}", "preço da vela fechada")
        + _caixa("STOP", f"{plano['stop']:.6g}", f"{plano['stop'] / e - 1:+.2%}")
        + _caixa("ALVO", f"{plano['alvo']:.6g}", f"{plano['alvo'] / e - 1:+.2%}") + "</div>"
        f"<div style='margin-top:10px;font-size:14px'>🕒 <b>Quando:</b> a vela {abre:%H:%M}–{fecha:%H:%M} "
        f"({'horário de Brasília' if FUSO == 'America/Sao_Paulo' else FUSO}) fechou há {max(idade, 0):.0f} min · entre "
        f"agora · saia no stop, no alvo ou até <b>{sair:%H:%M}</b> (≈{N_SAIDA} velas).</div>"
        f"<div style='margin-top:6px;font-size:13px'>💵 Exemplo (US$100 de margem, 10x = posição de US$1.000): alvo ≈ "
        f"<b>+US${g:.0f}</b> · stop ≈ <b>−US${p:.0f}</b> (já com taxas de {custo:.1%}). Perda no stop como % da margem: {alav}.</div>"
        f"{mercado_texto(i)}{aviso}</div>")
    mostrar(f"<p>{hist}{ml}</p>")
    fig = grafico(d, i, plano, f"{simbolo} {intervalo} — {titulo}")
    if png:
        fig.savefig(png, dpi=110, facecolor=fig.get_facecolor())
    plt.show()
    mostrar("<h4>🧠 Smart money (inferido de candles e volume)</h4><ul>"
            + "".join(f"<li>{a}</li>" for a in smc_texto(d, lado)) + "</ul>"
            "<h4>🔎 Análise gráfica</h4><ul>" + "".join(f"<li>{a}</li>" for a in razoes(d, lado)) + "</ul>"
            "<p style='font-size:12px;opacity:.75'>Smart money aqui é uma leitura heurística do preço e do volume "
            "(sem dados de carteiras, funding ou liquidações) e não é prova de ação institucional. Direção forçada: "
            "o app sempre escolhe o lado mais provável, mas sem vantagem comprovada o acerto esperado é ~50% e "
            "custos + alavancagem tornam o resultado esperado negativo; confira a 'confiança' e o acerto histórico. "
            "Estudo educacional, não é recomendação de investimento.</p>")
    return fig


def analisar(token="APT", intervalo="15m", paginas=20, sintetico=False, png=None):
    """Análise detalhada de UM token (sempre indica LONG ou SHORT)."""
    _config(intervalo)
    simbolo = normalizar(token)
    try:
        bruto, fonte = carregar(simbolo, intervalo, None, paginas, sintetico)
    except SystemExit as e:
        mostrar(f"<b>⚠️ {e}</b><br>Verifique o símbolo (ex.: APT, BTC, SOL) e tente de novo.")
        return
    d = indicadores(bruto)
    ctx = {} if sintetico else contexto(intervalo)
    mk, info = mercado(d, ctx, simbolo)
    i = direcao(d, modelo_dir(bruto), mk, info)
    return apresentar(simbolo, intervalo, fonte, bruto, d, i, png)


STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "USDE", "BUSD", "AEUR", "EUR",
           "XUSD", "USD1", "PAXG", "WBTC", "WBETH", "BFUSD"}
RESERVA = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT",
           "SUI", "APT", "NEAR", "ARB", "OP", "INJ", "TIA", "SEI", "LTC", "ATOM",
           "UNI", "AAVE", "FIL", "RUNE", "PEPE", "WIF", "FET", "TAO", "ENA", "TON",
           "TRX", "BCH", "ETC", "HBAR", "ICP", "POL", "ALGO", "STX", "IMX", "JUP",
           "ONDO", "PENDLE", "JTO", "LDO", "CRV", "MKR", "SAND", "MANA", "GALA", "AXS"]


def universo(n=60):
    """Tokens USDT líquidos, ordenados pelos que mais se moveram nas últimas 24h."""
    for base in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            j = _get(base + "/api/v3/ticker/24hr", {})
            L = [(x["symbol"], float(x["quoteVolume"]), abs(float(x["priceChangePercent"])))
                 for x in j if x["symbol"].endswith("USDT")
                 and x["symbol"][:-4] not in STABLES and float(x["quoteVolume"]) > 3e6]
            L = sorted(L, key=lambda t: -t[1])[:150]          # os 150 mais líquidos
            return [s for s, _, _ in sorted(L, key=lambda t: -t[2])][:n]  # mais ativos primeiro
        except Exception:
            continue
    return [b + "USDT" for b in RESERVA][:n]


def _status():
    try:
        from IPython.display import display, HTML
        h = display(HTML(""), display_id=True)
        return lambda t: h.update(HTML(f"<div style='font-size:15px;padding:6px'>{t}</div>"))
    except Exception:
        return lambda t: print(t, flush=True)


def varrer(intervalo="15m", max_tokens=60, mostrar_top=3, paginas=6):
    """Lê BTC/ETH, prevê LONG/SHORT de cada token e mostra os de maior força."""
    _config(intervalo)
    lista = universo(max_tokens)
    t0, itens = time.time(), []
    st = _status()
    st("📈 Lendo a tendência de BTC e ETH...")
    ctx = contexto(intervalo, paginas)
    if ctx.get("btc") is not None:
        b = ctx["btc"]["bias"].iloc[-1]
        e_ = ctx["eth"]["bias"].iloc[-1] if ctx.get("eth") is not None else 0
        mostrar(f"<p>📈 <b>Mercado agora ({intervalo}):</b> BTC {_seta(b)} ({b:+.2f}) · ETH {_seta(e_)} ({e_:+.2f}). "
                "Tokens que seguem BTC/ETH têm a direção ajustada por isso.</p>")
    for k, sym in enumerate(lista, 1):
        st(f"🔍 Fase 1/2 · analisando {k}/{len(lista)} · <b>{sym}</b>")
        try:
            bruto, fonte = carregar(sym, intervalo, None, paginas, False, True)
            d = indicadores(bruto)
            mk, info = mercado(d, ctx, sym)
            itens.append((sym, fonte, bruto, d, direcao(d, None, mk, info)))
        except (Exception, SystemExit):   # token sem dados suficientes: pula
            continue
    if not itens:
        mostrar("<b>⚠️ Não consegui baixar dados de nenhum token.</b>")
        return []
    itens.sort(key=lambda t: -chave(t[4]))
    top = []
    for k, (sym, fonte, bruto, d, i) in enumerate(itens[:8], 1):
        st(f"🤖 Fase 2/2 · modelo ML nos {min(8, len(itens))} mais fortes · {k}/{min(8, len(itens))} · <b>{sym}</b>")
        top.append((sym, fonte, bruto, d, direcao(d, modelo_dir(bruto), i["mk_ser"], i["info"])))
    top.sort(key=lambda t: -chave(t[4]))
    st(f"✅ {len(itens)} tokens analisados em {time.time() - t0:.0f}s · melhores abaixo")
    def _acerto(i):
        return f"{(i['r'] > 0).mean():.0%}" if len(i["r"]) else "—"
    tabela = "".join(
        f"<tr><td>#{n}</td><td><b>{s}</b></td><td>{'⬆️ LONG' if i['lado'] == 'long' else '⬇️ SHORT'}</td>"
        f"<td>{confianca(i)}</td><td>{_acerto(i)}</td>"
        f"<td>{'✅' if i['mk'] * i['sg'] > .1 else '⚠️' if i['mk'] * i['sg'] < -.1 else '➖'}</td></tr>"
        for n, (s, _, _, _, i) in enumerate(top, 1))
    mostrar("<h4>🏁 Ranking (do mais forte)</h4><table style='border-collapse:collapse;font-size:14px' cellpadding=5>"
            "<tr><th></th><th>Token</th><th>Direção</th><th>Confiança</th><th>Acerto hist.</th><th>BTC/ETH</th></tr>"
            + tabela + "</table>")
    for n, (sym, fonte, bruto, d, i) in enumerate(top[:mostrar_top], 1):
        apresentar(sym, intervalo, fonte, bruto, d, i, posicao=n)
    return [t[0] for t in top[:mostrar_top]]


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("token", nargs="?", default="VARRER")
    a.add_argument("--intervalo", default="15m")
    a.add_argument("--png", default="grafico.png")
    a.add_argument("--sintetico", action="store_true")
    x = a.parse_args()
    matplotlib.use("Agg")
    (varrer(x.intervalo) if x.token.upper() == "VARRER" else
     analisar(x.token, x.intervalo, sintetico=x.sintetico, png=x.png))
