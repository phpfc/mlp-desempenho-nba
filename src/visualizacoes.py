"""
Visualizações do relatório (matplotlib, apenas para desenhar os gráficos).
Gera os arquivos PNG em figuras/.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RESULT = os.path.join(RAIZ, "resultados")
DIR_DADOS = os.path.join(RAIZ, "dados", "processados")
DIR_FIG = os.path.join(RAIZ, "figuras")

# paleta (validada para daltonismo): série 1 azul, 2 laranja, 3 verde-água, 4 amarelo
AZUL, LARANJA, AQUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, SUAVE, GRADE, EIXO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": TINTA, "axes.labelcolor": TINTA2, "axes.edgecolor": EIXO, "axes.linewidth": 0.8,
    "xtick.color": SUAVE, "ytick.color": SUAVE, "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def salvar(fig, nome):
    fig.savefig(os.path.join(DIR_FIG, nome))
    plt.close(fig)
    print("  ", nome)


def ler_csv(caminho):
    with open(caminho) as f:
        return list(csv.DictReader(f))


def arquitetura_txt(cfg):
    return "-".join(map(str, cfg["camadas"]))


# ------------------------------------------------------------------ figuras
def fig_classes(prep):
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    grupos = ["Treino", "Teste"]
    bad = [prep["treino_total"] - prep["treino_good"], prep["teste_total"] - prep["teste_good"]]
    good = [prep["treino_good"], prep["teste_good"]]
    x = [0, 1]
    w = 0.28
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], bad, w, color=AZUL, label="Bad")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], good, w, color=LARANJA, label="Good")
    for barras in (b1, b2):
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, f"{int(b.get_height())}",
                    ha="center", va="bottom", fontsize=8, color=TINTA2)
    ax.set_xticks(x, grupos)
    ax.set_ylabel("Jogadores")
    ax.set_title("Distribuição das classes após a limpeza")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    salvar(fig, "fig01_classes.png")


def fig_pts_por_classe(limpo, limiar_pts):
    import random
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for classe, cor, yc, rotulo in ((0, AZUL, 1, "Bad"), (1, LARANJA, 0, "Good")):
        pts = [float(r["PTS"]) for r in limpo if int(r["alvo"]) == classe]
        ax.scatter(pts, [yc + rng.uniform(-0.28, 0.28) for _ in pts], s=12, color=cor, alpha=0.55,
                   linewidths=0, label=rotulo)
    ax.axvline(limiar_pts, color=TINTA2, linewidth=1)
    ax.text(limiar_pts + 0.4, 1.42, f"PTS = {limiar_pts:.1f}", color=TINTA2, fontsize=8, va="center")
    ax.set_yticks([0, 1], ["Good", "Bad"])
    ax.set_xlabel("Pontos por jogo (PTS)")
    ax.set_title("Pontos por jogo em cada classe")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, 1.6)
    salvar(fig, "fig02_pts_por_classe.png")


def fig_correlacao(estat):
    estat = sorted(estat, key=lambda r: float(r["correlacao_alvo"]))
    nomes = [r["atributo"] for r in estat]
    vals = [float(r["correlacao_alvo"]) for r in estat]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.barh(nomes, vals, height=0.55, color=AZUL)
    ax.axvline(0, color=EIXO, linewidth=0.8)
    ax.set_xlabel("Correlação de Pearson com o alvo (Good = 1)")
    ax.set_title("Correlação de cada atributo com o alvo")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    salvar(fig, "fig03_correlacao_alvo.png")


def fig_arquitetura(cfg, n_entradas):
    camadas = [n_entradas] + list(cfg["camadas"]) + [1]
    mostrar = [min(n, 9) for n in camadas]  # desenha no máximo 9 nós por camada
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_axis_off()
    xs = [i * 1.0 for i in range(len(camadas))]
    pos = []
    for x, n, m in zip(xs, camadas, mostrar):
        ys = [(j - (m - 1) / 2) * 0.32 for j in range(m)]
        pos.append(ys)
    # conexões
    for l in range(len(camadas) - 1):
        for y0 in pos[l]:
            for y1 in pos[l + 1]:
                ax.plot([xs[l], xs[l + 1]], [y0, y1], color=GRADE, linewidth=0.5, zorder=1)
    # nós
    for l, (x, ys) in enumerate(zip(xs, pos)):
        cor = AZUL if l == 0 else (AQUA if l < len(camadas) - 1 else LARANJA)
        for j, y in enumerate(ys):
            if camadas[l] > mostrar[l] and j == mostrar[l] // 2:
                ax.text(x, y, "⋮", ha="center", va="center", fontsize=12, color=TINTA2, zorder=3)
                continue
            ax.add_patch(Circle((x, y), 0.085, facecolor="white", edgecolor=cor, linewidth=1.4, zorder=2))
    rotulos = [f"Entrada\n{camadas[0]} atributos"] + \
              [f"Oculta {i + 1}\n{n} neurônios\n({cfg['ativacao']})" for i, n in enumerate(cfg["camadas"])] + \
              ["Saída\n1 neurônio\n(sigmoide)"]
    for x, r in zip(xs, rotulos):
        ax.text(x, -1.7, r, ha="center", va="top", fontsize=8, color=TINTA2)
    ax.set_xlim(-0.4, xs[-1] + 0.4)
    ax.set_ylim(-2.5, 1.6)
    ax.set_title(f"Arquitetura da MLP final: {n_entradas}-{arquitetura_txt(cfg)}-1")
    salvar(fig, "fig04_arquitetura.png")


def fig_grid(grid, pw_bal):
    arqs = ["8", "16", "32", "16-8"]
    taxas = [0.01, 0.1, 0.5]
    cores = [AZUL, LARANJA, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, pw, titulo in zip(axes, (1.0, pw_bal), ("Sem ponderação (pw = 1)", f"Classe positiva ponderada (pw = {pw_bal:.1f})")):
        for k, (taxa, cor) in enumerate(zip(taxas, cores)):
            medias, desvios = [], []
            for a in arqs:
                r = next(g for g in grid if arquitetura_txt(g["parametros"]) == a
                         and g["parametros"]["taxa"] == taxa and abs(g["parametros"]["peso_positivo"] - pw) < 1e-9)
                medias.append(r["resumo"]["f1"]["media"])
                desvios.append(r["resumo"]["f1"]["desvio"])
            xs = [i + (k - 1) * 0.2 for i in range(len(arqs))]
            ax.errorbar(xs, medias, yerr=desvios, fmt="o", color=cor, markersize=5, capsize=2,
                        elinewidth=1, label=f"taxa = {taxa}")
        ax.set_xticks(range(len(arqs)), [f"[{a}]" for a in arqs])
        ax.set_title(titulo)
        ax.set_xlabel("Neurônios nas camadas ocultas")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("F1 (classe Good) - média ± desvio, 5 folds")
    axes[0].legend(loc="lower left")
    fig.suptitle("Validação cruzada: arquitetura x taxa de aprendizado x ponderação", fontsize=10, fontweight="bold", color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    salvar(fig, "fig05_grid_cv.png")


def fig_curvas(melhor, epoca_escolhida):
    c = melhor["curvas"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax = axes[0]
    ax.plot(c["epoca"], c["perda_treino"], color=AZUL, linewidth=1.8, label="Treino")
    ax.plot(c["epoca"], c["perda_val"], color=LARANJA, linewidth=1.8, label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropia cruzada (ponderada)")
    ax.set_title("Perda média nos 5 folds")
    ax.legend()
    ax = axes[1]
    ax.plot(c["epoca"], c["f1_val"], color=LARANJA, linewidth=1.8, label="F1 validação")
    ax.plot(c["epoca"], c["acuracia_val"], color=AZUL, linewidth=1.8, label="Acurácia validação")
    ax.axvline(epoca_escolhida, color=TINTA2, linewidth=1)
    ax.text(epoca_escolhida + 4, ax.get_ylim()[0] + 0.02, f"{epoca_escolhida} épocas", fontsize=8, color=TINTA2)
    ax.set_xlabel("Época")
    ax.set_title("Métricas de validação por época")
    ax.legend(loc="center right")
    fig.suptitle(f"Curvas de treinamento - configuração {melhor['nome']}", fontsize=10, fontweight="bold", color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    salvar(fig, "fig06_curvas_treinamento.png")


def fig_matriz_confusao(m, nome="fig07_matriz_confusao.png", titulo="Matriz de confusão (teste, execução mediana)"):
    mat = [[m["VN"], m["FP"]], [m["FN"], m["VP"]]]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    maximo = max(max(l) for l in mat)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=maximo)
    for i in range(2):
        for j in range(2):
            v = mat[i][j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > maximo * 0.55 else TINTA)
    ax.set_xticks([0, 1], ["Bad", "Good"])
    ax.set_yticks([0, 1], ["Bad", "Good"])
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(titulo, fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    salvar(fig, nome)


def fig_roc(final, regra_pts):
    roc = final["mediana"]["roc"]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=EIXO, linewidth=0.8)
    ax.plot([p[0] for p in roc], [p[1] for p in roc], color=AZUL, linewidth=2,
            label=f"MLP (AUC = {final['mediana']['metricas']['auc']:.3f})")
    m = final["mediana"]["metricas"]
    ax.scatter([1 - m["especificidade"]], [m["recall"]], s=40, color=LARANJA, zorder=3, label="Limiar 0,5")
    ax.set_xlabel("Taxa de falsos positivos (1 - especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_title("Curva ROC no conjunto de teste")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    salvar(fig, "fig08_roc.png")


def fig_importancia(imp_final, imp_sem, n=12):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, imp, titulo in zip(axes, (imp_final, imp_sem), ("Modelo completo (31 atributos)", "Sem atributos de pontuação (22 atributos)")):
        top = sorted(imp, key=lambda r: -float(r["queda_auc"]))[:n][::-1]
        ax.barh([r["atributo"] for r in top], [float(r["queda_auc"]) for r in top], height=0.55, color=AZUL)
        ax.axvline(0, color=EIXO, linewidth=0.8)
        ax.set_xlabel("Queda de AUC ao permutar o atributo")
        ax.set_title(titulo)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Importância por permutação (conjunto de teste, média de 20 permutações)", fontsize=10, fontweight="bold", color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    salvar(fig, "fig09_importancia.png")


def fig_comparacao(exp):
    metricas = [("acuracia", "Acurácia"), ("precisao", "Precisão"), ("recall", "Recall"), ("especificidade", "Especif."), ("f1", "F1"), ("auc", "AUC")]
    series = [
        ("Classe majoritária", exp["baselines"]["majoritaria"], AMARELO),
        (f"Regra PTS ≥ {exp['baselines']['regra_pts']['limiar_pts']:.1f}", exp["baselines"]["regra_pts"]["metricas"], AQUA),
        ("MLP sem pontuação", {k: v["media"] for k, v in exp["cenario_sem_pontuacao"]["resumo"].items()}, LARANJA),
        ("MLP completa", {k: v["media"] for k, v in exp["final"]["resumo"].items()}, AZUL),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    w = 0.19
    for k, (rotulo, m, cor) in enumerate(series):
        xs = [i + (k - 1.5) * (w + 0.01) for i in range(len(metricas))]
        ax.bar(xs, [m[c] for c, _ in metricas], w, color=cor, label=rotulo)
    ax.set_xticks(range(len(metricas)), [r for _, r in metricas])
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Valor no conjunto de teste")
    ax.set_title("Comparação entre baselines e os dois cenários da MLP (média das 5 execuções)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=4)
    salvar(fig, "fig10_comparacao.png")


def main():
    os.makedirs(DIR_FIG, exist_ok=True)
    with open(os.path.join(DIR_RESULT, "experimentos.json")) as f:
        exp = json.load(f)
    with open(os.path.join(DIR_RESULT, "preparacao.json")) as f:
        prep = json.load(f)
    limpo = ler_csv(os.path.join(DIR_DADOS, "dataset_limpo.csv"))
    estat = ler_csv(os.path.join(DIR_RESULT, "estatisticas_descritivas.csv"))
    imp_final = ler_csv(os.path.join(DIR_RESULT, "importancia_final.csv"))
    imp_sem = ler_csv(os.path.join(DIR_RESULT, "importancia_sem_pontuacao.csv"))
    melhor = next(r for r in exp["ativacoes"] if r["nome"] == exp["melhor_ativacao"])

    print("gerando figuras:")
    fig_classes(prep)
    fig_pts_por_classe(limpo, exp["baselines"]["regra_pts"]["limiar_pts"])
    fig_correlacao(estat)
    fig_arquitetura(exp["config_final"], len(exp["colunas"]))
    fig_grid(exp["grid"], exp["peso_balanceado"])
    fig_curvas(melhor, exp["analise_epocas"]["epoca_escolhida"])
    fig_matriz_confusao(exp["final"]["mediana"]["metricas"])
    fig_matriz_confusao(exp["cenario_sem_pontuacao"]["mediana"]["metricas"], "fig07b_matriz_confusao_sem_pontuacao.png",
                        "Matriz de confusão - cenário sem pontuação")
    fig_roc(exp["final"], exp["baselines"]["regra_pts"])
    fig_importancia(imp_final, imp_sem)
    fig_comparacao(exp)


if __name__ == "__main__":
    main()
