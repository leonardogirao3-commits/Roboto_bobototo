"""
Bot de alertas: varre tokens e envia LONG/SHORT no Telegram (com gráfico).
Também acompanha cada sinal enviado e informa se bateu o ALVO, o STOP ou o TEMPO,
mantendo um placar real de acertos. Feito para rodar no GitHub Actions a cada 15 min.

Variáveis de ambiente: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID (obrigatórias para enviar);
INTERVALO (15m), MAX_TOKENS (40), MIN_CONF (fraca|media|forte; padrão media),
MAX_ALERTAS (2 por execução), COOLDOWN_MIN (120), FUSO (America/Sao_Paulo).
Uso: python alerta.py            -> execução normal
     python alerta.py --teste    -> envia mensagem de teste e confirma a fonte de dados
Estudo educacional, não é recomendação de investimento.
"""
import os, io, json, time, html, argparse
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import app_cripto as a

TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
INTERVALO = os.environ.get("INTERVALO", "15m")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "40"))
MAX_ALERTAS = int(os.environ.get("MAX_ALERTAS", "2"))
COOLDOWN_MIN = int(os.environ.get("COOLDOWN_MIN", "120"))
ESTADO = os.environ.get("ESTADO_ARQ", "estado/estado.json")
a.FUSO = os.environ.get("FUSO", "America/Sao_Paulo")
NIVEL = {"fraca": 0, "média": 1, "forte": 2}
MIN_NIVEL = NIVEL.get(os.environ.get("MIN_CONF", "media").lower().replace("media", "média"), 1)


def agora_utc():
    if os.environ.get("AGORA_TESTE"):            # gancho só para testes
        return pd.Timestamp(os.environ["AGORA_TESTE"])
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def fmt(x): return f"{x:.6g}"
def pc(v): return f"{v:+.2%}".replace(".", ",")
def fuso_nome(): return "Brasília" if a.FUSO == "America/Sao_Paulo" else a.FUSO


def tg(texto, foto=None):
    """Envia ao Telegram. Nunca imprime a URL (ela contém o token)."""
    if not (TOKEN and CHAT):
        print("[sem Telegram configurado]", texto.replace("\n", " | ")[:300])
        return False
    base = f"https://api.telegram.org/bot{TOKEN}"
    try:
        if foto is not None:
            r = requests.post(base + "/sendPhoto", timeout=30,
                              data={"chat_id": CHAT, "caption": texto[:1000], "parse_mode": "HTML"},
                              files={"photo": ("grafico.png", foto, "image/png")})
        else:
            r = requests.post(base + "/sendMessage", timeout=30,
                              data={"chat_id": CHAT, "text": texto, "parse_mode": "HTML"})
        r.raise_for_status()
        return True
    except Exception as e:
        code = getattr(getattr(e, "response", None), "status_code", "")
        print(f"Falha ao enviar ao Telegram: {type(e).__name__} {code}")
        return False


# ------------------------------------------------------------------ estado
def carregar_estado():
    try:
        with open(ESTADO, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"pendentes": [], "enviados": {}, "placar": {"n": 0, "ganhos": 0, "soma": 0.0}, "erro_em": None}


def salvar_estado(e):
    os.makedirs(os.path.dirname(ESTADO) or ".", exist_ok=True)
    with open(ESTADO, "w", encoding="utf-8") as f:
        json.dump(e, f, indent=1)


# ------------------------------------------------- acompanhar sinais enviados
def recentes(sym, intervalo):
    for nome, fn in sorted(a.FONTES[:5], key=lambda f: f[0] != a._PREF):
        try:
            df = fn(sym, intervalo, 1)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            continue
    return None


def resolver(estado, agora):
    """Confere cada sinal pendente: ALVO, STOP ou TEMPO (stop primeiro se ambos na mesma vela)."""
    ainda = []
    for s in estado["pendentes"]:
        df = recentes(s["sym"], s["intervalo"])
        t0, t_exp = pd.Timestamp(s["t_sinal"]), pd.Timestamp(s["t_exp"])
        m = a.MINUTOS[s["intervalo"]]
        if df is None:
            ainda.append(s)
            continue
        fut = df[(df.index >= t0 + pd.Timedelta(minutes=m)) & (df.index < t_exp)]
        sg = 1 if s["lado"] == "long" else -1
        saida, motivo = None, None
        for _, v in fut.iterrows():
            stop_hit = v["Low"] <= s["stop"] if sg == 1 else v["High"] >= s["stop"]
            alvo_hit = v["High"] >= s["alvo"] if sg == 1 else v["Low"] <= s["alvo"]
            if stop_hit:
                saida, motivo = s["stop"], "STOP"
                break
            if alvo_hit:
                saida, motivo = s["alvo"], "ALVO"
                break
        if saida is None and agora >= t_exp and len(fut):
            saida, motivo = float(fut["Close"].iloc[-1]), "TEMPO"
        if saida is None:
            ainda.append(s)
            continue
        ret = sg * (saida / s["entrada"] - 1) - 2 * a.CUSTO
        p = estado["placar"]
        p["n"] += 1
        p["ganhos"] += int(ret > 0)
        p["soma"] += ret
        ic = "✅" if ret > 0 else "❌"
        tg(f"{ic} <b>{motivo}</b> · {html.escape(s['sym'])} {s['lado'].upper()}: <b>{pc(ret)}</b> líquido (com taxas)\n"
           f"Placar real dos alertas: {p['ganhos']} de {p['n']} ({p['ganhos'] / p['n']:.0%}) · "
           f"média {pc(p['soma'] / p['n'])} por operação")
    estado["pendentes"] = ainda


# --------------------------------------------------------------- varredura
def varrer_sinais():
    a._config(INTERVALO)
    lista = a.universo(MAX_TOKENS)
    ctx = a.contexto(INTERVALO, 6)
    itens = []
    for sym in lista:
        try:
            bruto, fonte = a.carregar(sym, INTERVALO, None, 6, False, True)
            d = a.indicadores(bruto)
            mk, info = a.mercado(d, ctx, sym)
            itens.append((sym, fonte, bruto, d, a.direcao(d, None, mk, info)))
        except (Exception, SystemExit):
            continue
    itens.sort(key=lambda t: -a.chave(t[4]))
    top = []
    for sym, fonte, bruto, d, i in itens[:8]:
        try:
            top.append((sym, fonte, bruto, d, a.direcao(d, a.modelo_dir(bruto), i["mk_ser"], i["info"])))
        except Exception:
            continue
    top.sort(key=lambda t: -a.chave(t[4]))
    return top, len(itens)


def montar(sym, d, i, plano, conf):
    long_ = i["lado"] == "long"
    m = a.MINUTOS[INTERVALO]
    fecha = a.local(d.index[-1]) + pd.Timedelta(minutes=m)
    sair = fecha + pd.Timedelta(minutes=m * a.N_SAIDA)
    adj = i["mk"] * i["sg"]
    mercado = "a favor" if adj > .1 else "CONTRA" if adj < -.1 else "neutro"
    r, e = i["r"], plano["entrada"]
    hist = (f"Histórico do token (n={len(r)}): acerto {(r > 0).mean():.0%} · empate em {i['be']:.0%}"
            if len(r) else "Histórico: amostra insuficiente")
    smc = [x for x in a.smc_texto(d, i["lado"]) if not x.startswith("Sem sinal")]
    L = [f"{'⬆️' if long_ else '⬇️'} <b>{'LONG' if long_ else 'SHORT'}</b> · {html.escape(sym)} · {INTERVALO}",
         f"Confiança: <b>{conf}</b> · BTC/ETH: {mercado}", "",
         f"Entrada ≈ <b>{fmt(e)}</b>",
         f"Stop <b>{fmt(plano['stop'])}</b> ({pc(plano['stop'] / e - 1)})",
         f"Alvo <b>{fmt(plano['alvo'])}</b> ({pc(plano['alvo'] / e - 1)})", "",
         f"Vela fechou {fecha:%H:%M} ({fuso_nome()}) · saia até {sair:%H:%M}", hist,
         f"Com 10x, o stop custa {plano['sp'] * 10:.0%} da margem"]
    if smc:
        L.append("🧠 " + html.escape(smc[0]))
    L.append("<i>Estudo educacional; não é recomendação de investimento.</i>")
    return "\n".join(L)


def executar():
    estado = carregar_estado()
    agora = agora_utc()
    resolver(estado, agora)
    top, lidos = varrer_sinais()
    if lidos == 0:                                    # nenhuma fonte respondeu
        ult = pd.Timestamp(estado["erro_em"]) if estado.get("erro_em") else None
        if ult is None or (agora - ult) > pd.Timedelta(hours=6):
            tg("⚠️ Bot de alertas: não consegui baixar dados de nenhuma exchange nesta execução.")
            estado["erro_em"] = agora.isoformat()
        salvar_estado(estado)
        return
    enviados = 0
    m = a.MINUTOS[INTERVALO]
    for sym, fonte, bruto, d, i in top:
        if enviados >= MAX_ALERTAS:
            break
        conf = a.confianca(i)
        if NIVEL[conf] < MIN_NIVEL:
            continue
        idade = (agora - (d.index[-1] + pd.Timedelta(minutes=m))).total_seconds() / 60
        if idade > 2 * m:                             # dados velhos: não alertar
            continue
        ult = estado["enviados"].get(sym)
        if (ult and (agora - pd.Timestamp(ult)) < pd.Timedelta(minutes=COOLDOWN_MIN)) or \
                any(s["sym"] == sym for s in estado["pendentes"]):
            continue
        u = d.iloc[-1]
        plano = a.niveis(u["c"], u["atr"], i["lado"])
        fig = a.grafico(d, i, plano, f"{sym} {INTERVALO} — {'LONG' if i['lado'] == 'long' else 'SHORT'}")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=90, facecolor=fig.get_facecolor())
        plt.close(fig)
        if tg(montar(sym, d, i, plano, conf), buf.getvalue()):
            enviados += 1
            t0 = d.index[-1]
            estado["enviados"][sym] = agora.isoformat()
            estado["pendentes"].append(dict(
                sym=sym, lado=i["lado"], intervalo=INTERVALO, t_sinal=t0.isoformat(),
                t_exp=(t0 + pd.Timedelta(minutes=m * (a.N_SAIDA + 1))).isoformat(),
                entrada=float(plano["entrada"]), stop=float(plano["stop"]), alvo=float(plano["alvo"])))
    estado["enviados"] = {k: v for k, v in estado["enviados"].items()
                          if (agora - pd.Timestamp(v)) < pd.Timedelta(hours=24)}
    salvar_estado(estado)
    print(f"Execução ok: {lidos} tokens analisados, {enviados} alerta(s) enviado(s), "
          f"{len(estado['pendentes'])} sinal(is) em acompanhamento.")


def teste():
    try:
        bruto, fonte = a.carregar("BTCUSDT", INTERVALO, None, 2, False, True)
        ult = a.local(bruto.index[-1])
        msg = (f"✅ <b>Bot conectado.</b>\nFonte de dados: {fonte}\n"
               f"Última vela fechada de BTC: {ult:%d/%m %H:%M} ({fuso_nome()})\n"
               f"Configuração: {INTERVALO} · {MAX_TOKENS} tokens · confiança mínima "
               f"{[k for k, v in NIVEL.items() if v == MIN_NIVEL][0]}")
    except (Exception, SystemExit) as e:
        msg = f"⚠️ Telegram OK, mas não consegui baixar dados de preço ({type(e).__name__})."
    print(msg.replace("\n", " | "))
    tg(msg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teste", action="store_true")
    teste() if p.parse_args().teste else executar()
