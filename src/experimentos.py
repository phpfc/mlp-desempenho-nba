"""
Etapas 3 e 4 - Treinamento, ajuste de hiperparâmetros e avaliação.

Protocolo:
  A) Validação cruzada estratificada (5 folds) no conjunto de TREINO para
     comparar arquiteturas x taxas de aprendizado x ponderação da classe
     positiva (função de ativação sigmoide).
  B) Com a melhor combinação de A, compara as funções de ativação
     (sigmoide, tanh, relu) pelo mesmo protocolo.
  C) Analisa o número de épocas a partir das curvas de validação da melhor
     configuração e escolhe o número de épocas.
  D) Executa a configuração final N vezes (sementes diferentes) treinando com
     TODO o conjunto de treino e avaliando no conjunto de TESTE (nunca usado
     antes). Guarda a execução mediana (modelo, probabilidades, curva ROC).
  E) Baselines: classe majoritária e regra "PTS >= limiar" (limiar escolhido
     no treino).
  F) Cenário alternativo: mesma configuração, sem os atributos de volume de
     pontuação (PTS, FG, FGA, 3P, 3PA, 2P, 2PA, FT, FTA).

Resultados em resultados/experimentos.json (+ modelos em JSON).
"""
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MLP  # noqa: E402
from metricas import calcular_metricas, curva_roc, media, desvio_padrao  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DADOS = os.path.join(RAIZ, "dados", "processados")
DIR_RESULT = os.path.join(RAIZ, "resultados")

SEMENTE = 42
K_FOLDS = 5
EPOCAS_GRID = 300
LOTE = 32
AVALIAR_A_CADA = 10
N_EXECUCOES = 5
N_PROCESSOS = 8

ARQUITETURAS = [[8], [16], [32], [16, 8]]
TAXAS = [0.01, 0.1, 0.5]
ATIVACOES = ["sigmoide", "tanh", "relu"]
ATRIBUTOS_VOLUME = ["PTS", "FG", "FGA", "3P", "3PA", "2P", "2PA", "FT", "FTA"]


# ------------------------------------------------------------------ dados
def ler_csv(caminho, excluir=()):
    with open(caminho) as f:
        linhas = list(csv.DictReader(f))
    colunas = [c for c in linhas[0] if c not in ("Player", "alvo") and c not in excluir]
    X = [[float(l[c]) for c in colunas] for l in linhas]
    y = [int(l["alvo"]) for l in linhas]
    nomes = [l["Player"] for l in linhas]
    return X, y, colunas, nomes


def folds_estratificados(y, k, semente):
    rng = random.Random(semente)
    folds = [[] for _ in range(k)]
    for classe in (0, 1):
        idx = [i for i, v in enumerate(y) if v == classe]
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            folds[n % k].append(i)
    return folds


def nome_config(c):
    return f"{'-'.join(map(str, c['camadas']))} | taxa={c['taxa']} | pw={c['peso_positivo']:.1f} | {c['ativacao']}"


# ---------------------------------------------------------- um fold de CV
def executar_fold(args):
    config, fold_id, folds, X, y = args
    val = set(folds[fold_id])
    Xtr = [x for i, x in enumerate(X) if i not in val]
    ytr = [v for i, v in enumerate(y) if i not in val]
    Xval = [X[i] for i in folds[fold_id]]
    yval = [y[i] for i in folds[fold_id]]
    t0 = time.time()
    rede = MLP(len(X[0]), config["camadas"], config["ativacao"], semente=SEMENTE + fold_id)
    hist = rede.treinar(Xtr, ytr, epocas=config["epocas"], taxa=config["taxa"], tamanho_lote=LOTE,
                        peso_positivo=config["peso_positivo"], X_val=Xval, y_val=yval,
                        avaliar_a_cada=AVALIAR_A_CADA)
    m = calcular_metricas(yval, rede.prever(Xval))
    _, m["auc"] = curva_roc(yval, rede.prever_proba(Xval))
    return {"config": nome_config(config), "fold": fold_id, "metricas": m,
            "historico": hist, "tempo": time.time() - t0}


def validacao_cruzada(configs, folds, X, y, executor):
    tarefas = [(c, k, folds, X, y) for c in configs for k in range(K_FOLDS)]
    saidas = list(executor.map(executar_fold, tarefas))
    resultados = []
    for c in configs:
        nome = nome_config(c)
        fs = [s for s in saidas if s["config"] == nome]
        resumo = {}
        for chave in ("acuracia", "erro", "precisao", "recall", "especificidade", "f1", "auc"):
            vals = [f["metricas"][chave] for f in fs]
            resumo[chave] = {"media": media(vals), "desvio": desvio_padrao(vals)}
        epocas = fs[0]["historico"]["epoca"]
        curvas = {}
        for chave in ("perda_treino", "perda_val", "f1_val", "acuracia_val"):
            curvas[chave] = [media([f["historico"][chave][i] for f in fs]) for i in range(len(epocas))]
        resultados.append({
            "nome": nome, "parametros": c, "resumo": resumo,
            "folds": [f["metricas"] for f in fs],
            "curvas": {"epoca": epocas, **curvas},
            "tempo_medio_s": media([f["tempo"] for f in fs]),
        })
    return resultados


def melhor(resultados):
    return max(resultados, key=lambda r: (r["resumo"]["f1"]["media"], r["resumo"]["auc"]["media"]))


# ------------------------------------------------------- execuções finais
def executar_final(args):
    config, semente, Xtr, ytr, Xte, yte = args
    t0 = time.time()
    rede = MLP(len(Xtr[0]), config["camadas"], config["ativacao"], semente=semente)
    hist = rede.treinar(Xtr, ytr, epocas=config["epocas"], taxa=config["taxa"], tamanho_lote=LOTE,
                        peso_positivo=config["peso_positivo"], avaliar_a_cada=AVALIAR_A_CADA)
    probas = rede.prever_proba(Xte)
    pred = [1 if p >= 0.5 else 0 for p in probas]
    m = calcular_metricas(yte, pred)
    roc, m["auc"] = curva_roc(yte, probas)
    return {"semente": semente, "metricas": m, "probabilidades": probas, "roc": roc,
            "historico": hist, "modelo": rede.exportar(), "tempo": time.time() - t0}


def resumir_execucoes(execucoes):
    resumo = {}
    for chave in ("acuracia", "erro", "precisao", "recall", "especificidade", "f1", "auc"):
        vals = [e["metricas"][chave] for e in execucoes]
        resumo[chave] = {"media": media(vals), "desvio": desvio_padrao(vals), "min": min(vals), "max": max(vals)}
    return resumo


def execucao_mediana(execucoes):
    ordenadas = sorted(execucoes, key=lambda e: (e["metricas"]["f1"], e["metricas"]["auc"]))
    return ordenadas[len(ordenadas) // 2]


def rodar_cenario(config, Xtr, ytr, Xte, yte, nomes_te, executor):
    tarefas = [(config, s, Xtr, ytr, Xte, yte) for s in range(N_EXECUCOES)]
    execucoes = list(executor.map(executar_final, tarefas))
    med = execucao_mediana(execucoes)
    erros = [{"jogador": nomes_te[i], "real": yte[i], "prob": med["probabilidades"][i]}
             for i in range(len(yte)) if (med["probabilidades"][i] >= 0.5) != (yte[i] == 1)]
    return {
        "config": config,
        "execucoes": [{"semente": e["semente"], "metricas": e["metricas"], "tempo": e["tempo"],
                       "perda_treino": e["historico"]["perda_treino"],
                       "epoca": e["historico"]["epoca"]} for e in execucoes],
        "resumo": resumir_execucoes(execucoes),
        "mediana": {"semente": med["semente"], "metricas": med["metricas"], "roc": med["roc"],
                    "probabilidades": med["probabilidades"], "erros": erros},
        "modelo_mediana": med["modelo"],
    }


# --------------------------------------------------------------- baselines
def baseline_majoritaria(yte):
    m = calcular_metricas(yte, [0] * len(yte))
    m["auc"] = 0.5
    return m


def baseline_regra_pts(Xtr, ytr, Xte, yte, colunas):
    """Regra 'Good se PTS >= limiar'; limiar escolhido maximizando o F1 no treino."""
    i = colunas.index("PTS")
    valores = sorted(set(x[i] for x in Xtr))
    candidatos = [(a + b) / 2 for a, b in zip(valores, valores[1:])]
    melhor_lim, melhor_f1 = None, -1
    for lim in candidatos:
        f1 = calcular_metricas(ytr, [1 if x[i] >= lim else 0 for x in Xtr])["f1"]
        if f1 > melhor_f1:
            melhor_lim, melhor_f1 = lim, f1
    pred = [1 if x[i] >= melhor_lim else 0 for x in Xte]
    m = calcular_metricas(yte, pred)
    _, m["auc"] = curva_roc(yte, [x[i] for x in Xte])
    with open(os.path.join(DIR_DADOS, "padronizacao.json")) as f:
        p = json.load(f)["parametros"]["PTS"]
    return {"limiar_padronizado": melhor_lim, "limiar_pts": melhor_lim * p["desvio"] + p["media"],
            "f1_treino": melhor_f1, "metricas": m}


# -------------------------------------------------------------------- main
def main():
    os.makedirs(DIR_RESULT, exist_ok=True)
    t_inicio = time.time()
    Xtr, ytr, colunas, _ = ler_csv(os.path.join(DIR_DADOS, "treino.csv"))
    Xte, yte, _, nomes_te = ler_csv(os.path.join(DIR_DADOS, "teste.csv"))
    n_pos = sum(ytr)
    pw_balanceado = (len(ytr) - n_pos) / n_pos
    print(f"treino: {len(ytr)} ({n_pos} Good)  teste: {len(yte)} ({sum(yte)} Good)  atributos: {len(colunas)}")
    print(f"peso balanceado da classe positiva: {pw_balanceado:.2f}")
    folds = folds_estratificados(ytr, K_FOLDS, SEMENTE)
    saida = {"colunas": colunas, "n_treino": len(ytr), "n_teste": len(yte), "folds": K_FOLDS,
             "epocas_grid": EPOCAS_GRID, "lote": LOTE, "peso_balanceado": pw_balanceado}

    with ProcessPoolExecutor(max_workers=N_PROCESSOS) as executor:
        # ---- A) grid arquitetura x taxa x ponderação
        t0 = time.time()
        grid = [{"camadas": a, "taxa": t, "peso_positivo": pw, "ativacao": "sigmoide", "epocas": EPOCAS_GRID}
                for a in ARQUITETURAS for t in TAXAS for pw in (1.0, pw_balanceado)]
        saida["grid"] = validacao_cruzada(grid, folds, Xtr, ytr, executor)
        print(f"\n[A] grid ({len(grid)} configs x {K_FOLDS} folds) em {time.time() - t0:.0f}s")
        for r in sorted(saida["grid"], key=lambda r: -r["resumo"]["f1"]["media"]):
            s = r["resumo"]
            print(f"  {r['nome']:<40} f1={s['f1']['media']:.3f}±{s['f1']['desvio']:.3f} "
                  f"acc={s['acuracia']['media']:.3f} rec={s['recall']['media']:.3f} "
                  f"prec={s['precisao']['media']:.3f} auc={s['auc']['media']:.3f}")
        melhor_a = melhor(saida["grid"])
        saida["melhor_grid"] = melhor_a["nome"]

        # ---- B) funções de ativação
        t0 = time.time()
        base = melhor_a["parametros"]
        configs_b = [dict(base, ativacao=a) for a in ATIVACOES if a != "sigmoide"]
        saida["ativacoes"] = [melhor_a] + validacao_cruzada(configs_b, folds, Xtr, ytr, executor)
        print(f"\n[B] ativações em {time.time() - t0:.0f}s")
        for r in saida["ativacoes"]:
            s = r["resumo"]
            print(f"  {r['nome']:<40} f1={s['f1']['media']:.3f}±{s['f1']['desvio']:.3f} "
                  f"acc={s['acuracia']['media']:.3f} auc={s['auc']['media']:.3f}")
        melhor_b = melhor(saida["ativacoes"])
        saida["melhor_ativacao"] = melhor_b["nome"]

        # ---- C) número de épocas (curvas médias de validação da melhor configuração)
        curvas = melhor_b["curvas"]
        candidatos = list(zip(curvas["epoca"], curvas["f1_val"], curvas["perda_val"], curvas["acuracia_val"]))
        ep_escolhida = max(candidatos, key=lambda c: (round(c[1], 4), -c[2]))[0]
        saida["analise_epocas"] = {"config": melhor_b["nome"], "epoca_escolhida": ep_escolhida,
                                   "tabela": [{"epoca": e, "f1_val": f, "perda_val": p, "acuracia_val": a}
                                              for e, f, p, a in candidatos]}
        print(f"\n[C] épocas escolhidas: {ep_escolhida}")

        # ---- D) execuções finais no conjunto de teste
        t0 = time.time()
        config_final = dict(melhor_b["parametros"], epocas=ep_escolhida)
        saida["config_final"] = config_final
        saida["final"] = rodar_cenario(config_final, Xtr, ytr, Xte, yte, nomes_te, executor)
        print(f"\n[D] {N_EXECUCOES} execuções finais em {time.time() - t0:.0f}s")
        for e in saida["final"]["execucoes"]:
            m = e["metricas"]
            print(f"  semente {e['semente']}: acc={m['acuracia']:.3f} prec={m['precisao']:.3f} "
                  f"rec={m['recall']:.3f} esp={m['especificidade']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f} "
                  f"VP={m['VP']} FP={m['FP']} VN={m['VN']} FN={m['FN']}")
        print("  erros da execução mediana:", saida["final"]["mediana"]["erros"])

        # ---- E) baselines
        saida["baselines"] = {"majoritaria": baseline_majoritaria(yte),
                              "regra_pts": baseline_regra_pts(Xtr, ytr, Xte, yte, colunas)}
        print(f"\n[E] baseline majoritária acc={saida['baselines']['majoritaria']['acuracia']:.3f}; "
              f"regra PTS >= {saida['baselines']['regra_pts']['limiar_pts']:.1f}: "
              f"{ {k: round(v, 3) for k, v in saida['baselines']['regra_pts']['metricas'].items()} }")

        # ---- F) cenário sem atributos de volume de pontuação
        t0 = time.time()
        Xtr_b, ytr_b, colunas_b, _ = ler_csv(os.path.join(DIR_DADOS, "treino.csv"), excluir=ATRIBUTOS_VOLUME)
        Xte_b, yte_b, _, _ = ler_csv(os.path.join(DIR_DADOS, "teste.csv"), excluir=ATRIBUTOS_VOLUME)
        saida["cenario_sem_pontuacao"] = rodar_cenario(config_final, Xtr_b, ytr_b, Xte_b, yte_b, nomes_te, executor)
        saida["cenario_sem_pontuacao"]["colunas"] = colunas_b
        s = saida["cenario_sem_pontuacao"]["resumo"]
        print(f"\n[F] sem pontuação ({len(colunas_b)} atributos) em {time.time() - t0:.0f}s: "
              f"acc={s['acuracia']['media']:.3f} prec={s['precisao']['media']:.3f} rec={s['recall']['media']:.3f} "
              f"f1={s['f1']['media']:.3f} auc={s['auc']['media']:.3f}")

    saida["tempo_total_s"] = time.time() - t_inicio
    # modelos salvos à parte (arquivo principal fica menor)
    for chave, arq in (("final", "modelo_final.json"), ("cenario_sem_pontuacao", "modelo_sem_pontuacao.json")):
        with open(os.path.join(DIR_RESULT, arq), "w") as f:
            json.dump(saida[chave].pop("modelo_mediana"), f)
    with open(os.path.join(DIR_RESULT, "experimentos.json"), "w") as f:
        json.dump(saida, f, indent=1)
    print(f"\nconcluído em {saida['tempo_total_s']:.0f}s")


if __name__ == "__main__":
    main()
