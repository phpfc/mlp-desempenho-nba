"""Gera os fragmentos de tabelas LaTeX do relatório (relatorio/tabelas/*.tex)."""
import csv
import json
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_RESULT = os.path.join(RAIZ, "resultados")
DIR_DADOS = os.path.join(RAIZ, "dados", "processados")
DIR_TAB = os.path.join(RAIZ, "relatorio", "tabelas")


def esc(s):
    return str(s).replace("%", r"\%").replace("_", r"\_").replace("&", r"\&").replace("≥", r"$\geq$")


def f3(v):
    return f"{v:.3f}".replace(".", ",")


def f1(v):
    return f"{v:.1f}".replace(".", ",")


def pm(d):
    return f"{f3(d['media'])} $\\pm$ {f3(d['desvio'])}"


def escrever(nome, linhas):
    with open(os.path.join(DIR_TAB, nome), "w") as f:
        f.write("\n".join(linhas) + "\n")
    print("  ", nome)


def ler_csv(c):
    with open(c) as f:
        return list(csv.DictReader(f))


def arq(c):
    return "{[" + "-".join(map(str, c["camadas"])) + "]}"


def main():
    os.makedirs(DIR_TAB, exist_ok=True)
    exp = json.load(open(os.path.join(DIR_RESULT, "experimentos.json")))
    prep = json.load(open(os.path.join(DIR_RESULT, "preparacao.json")))
    estat = ler_csv(os.path.join(DIR_RESULT, "estatisticas_descritivas.csv"))
    limpo = {r["Player"]: r for r in ler_csv(os.path.join(DIR_DADOS, "dataset_limpo.csv"))}
    imp_final = ler_csv(os.path.join(DIR_RESULT, "importancia_final.csv"))
    imp_sem = ler_csv(os.path.join(DIR_RESULT, "importancia_sem_pontuacao.csv"))

    # --- preparação
    l = [r"\begin{tabular}{lr}", r"\toprule", r"Etapa & Linhas \\", r"\midrule",
         f"Dataset bruto & {prep['linhas_brutas']} \\\\",
         f"Linhas sem rótulo removidas & $-${prep['linhas_sem_alvo']} \\\\",
         f"Linhas por time de {prep["jogadores_trocados"]} jogadores trocados (mantida a linha TOT) & $-${prep['linhas_por_time_removidas']} \\\\",
         f"\\textbf{{Dataset limpo}} & \\textbf{{{prep['linhas_limpas']}}} \\\\",
         f"\\quad classe Good & {prep['classe_good']} \\\\",
         f"\\quad classe Bad & {prep['classe_bad']} \\\\",
         r"\midrule",
         f"Treino (70\\%) & {prep['treino_total']} ({prep['treino_good']} Good) \\\\",
         f"Teste (30\\%) & {prep['teste_total']} ({prep['teste_good']} Good) \\\\",
         r"\bottomrule", r"\end{tabular}"]
    escrever("preparacao.tex", l)

    # --- faltantes
    l = [r"\begin{tabular}{lrl}", r"\toprule", r"Coluna & Faltantes & Causa / tratamento \\", r"\midrule"]
    for col, tent in (("FG%", "FGA"), ("3P%", "3PA"), ("2P%", "2PA"), ("FT%", "FTA"), ("eFG%", "FGA")):
        l.append(f"{esc(col)} & {prep['faltantes_' + col]} & {esc(tent)} = 0 (divisão 0/0); preenchido com 0 \\\\")
    l += [r"\bottomrule", r"\end{tabular}"]
    escrever("faltantes.tex", l)

    # --- estatísticas descritivas
    l = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Atributo & Média Bad & Média Good & Desvio padrão & Correlação com o alvo \\", r"\midrule"]
    for r in estat:
        l.append(f"{esc(r['atributo'])} & {f3(float(r['media_bad']))} & {f3(float(r['media_good']))} & {f3(float(r['desvio_geral']))} & {f3(float(r['correlacao_alvo']))} \\\\")
    l += [r"\bottomrule", r"\end{tabular}"]
    escrever("estatisticas.tex", l)

    # --- grid CV
    l = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Camadas ocultas & Taxa & $w_+$ & Acurácia & Precisão & Recall & Especif. & F1 & AUC \\", r"\midrule"]
    for r in sorted(exp["grid"], key=lambda r: -r["resumo"]["f1"]["media"]):
        p, s = r["parametros"], r["resumo"]
        marca = r"\textbf{" if r["nome"] == exp["melhor_grid"] else ""
        fim = "}" if marca else ""
        l.append(f"{marca}{arq(p)}{fim} & {str(p['taxa']).replace('.', ',')} & {f1(p['peso_positivo'])} & {f3(s['acuracia']['media'])} & "
                 f"{f3(s['precisao']['media'])} & {f3(s['recall']['media'])} & {f3(s['especificidade']['media'])} & "
                 f"{marca}{pm(s['f1'])}{fim} & {f3(s['auc']['media'])} \\\\")
    l += [r"\bottomrule", r"\end{tabular}"]
    escrever("grid.tex", l)

    # --- ativações
    l = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Ativação & Acurácia & Precisão & Recall & Especif. & F1 & AUC \\", r"\midrule"]
    for r in exp["ativacoes"]:
        p, s = r["parametros"], r["resumo"]
        l.append(f"{p['ativacao']} & {f3(s['acuracia']['media'])} & {f3(s['precisao']['media'])} & {f3(s['recall']['media'])} & "
                 f"{f3(s['especificidade']['media'])} & {pm(s['f1'])} & {f3(s['auc']['media'])} \\\\")
    l += [r"\bottomrule", r"\end{tabular}"]
    escrever("ativacoes.tex", l)

    # --- épocas
    mostrar = set(list(range(10, 160, 10)) + [200, 250, 300])
    l = [r"\begin{tabular}{rrrr}", r"\toprule", r"Épocas & Perda validação & Acurácia validação & F1 validação \\", r"\midrule"]
    for t in exp["analise_epocas"]["tabela"]:
        if t["epoca"] in mostrar:
            marca = r"\textbf{" if t["epoca"] == exp["analise_epocas"]["epoca_escolhida"] else ""
            fim = "}" if marca else ""
            l.append(f"{marca}{t['epoca']}{fim} & {f3(t['perda_val'])} & {f3(t['acuracia_val'])} & {marca}{f3(t['f1_val'])}{fim} \\\\")
    l += [r"\bottomrule", r"\end{tabular}"]
    escrever("epocas.tex", l)

    # --- execuções finais (cenário completo e sem pontuação)
    for chave, nome in (("final", "execucoes.tex"), ("cenario_sem_pontuacao", "execucoes_sem_pontuacao.tex")):
        c = exp[chave]
        l = [r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule",
             r"Execução & VP & FP & VN & FN & Acurácia & Erro & Precisão & Recall & Especif. & F1 & AUC \\", r"\midrule"]
        for e in c["execucoes"]:
            m = e["metricas"]
            med = " (mediana)" if e["semente"] == c["mediana"]["semente"] else ""
            l.append(f"semente {e['semente']}{med} & {m['VP']} & {m['FP']} & {m['VN']} & {m['FN']} & {f3(m['acuracia'])} & {f3(m['erro'])} & "
                     f"{f3(m['precisao'])} & {f3(m['recall'])} & {f3(m['especificidade'])} & {f3(m['f1'])} & {f3(m['auc'])} \\\\")
        s = c["resumo"]
        l.append(r"\midrule")
        l.append(f"\\textbf{{Média $\\pm$ desvio}} & & & & & {pm(s['acuracia'])} & {pm(s['erro'])} & {pm(s['precisao'])} & "
                 f"{pm(s['recall'])} & {pm(s['especificidade'])} & {pm(s['f1'])} & {pm(s['auc'])} \\\\")
        l += [r"\bottomrule", r"\end{tabular}"]
        escrever(nome, l)

    # --- comparação
    rp = exp["baselines"]["regra_pts"]
    linhas = [("Classe majoritária (sempre Bad)", exp["baselines"]["majoritaria"]),
              (f"Regra PTS $\\geq$ {f1(rp['limiar_pts'])}", rp["metricas"]),
              ("MLP sem atributos de pontuação (média de 5)", {k: v["media"] for k, v in exp["cenario_sem_pontuacao"]["resumo"].items()}),
              ("MLP completa (média de 5)", {k: v["media"] for k, v in exp["final"]["resumo"].items()})]
    l = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Modelo & Acurácia & Precisão & Recall & Especif. & F1 & AUC \\", r"\midrule"]
    for nome, m in linhas:
        l.append(f"{nome} & {f3(m['acuracia'])} & {f3(m['precisao'])} & {f3(m['recall'])} & {f3(m['especificidade'])} & {f3(m['f1'])} & {f3(m['auc'])} \\\\")
    l += [r"\bottomrule", r"\end{tabular}"]
    escrever("comparacao.tex", l)

    # --- importância
    for imp, nome in ((imp_final, "importancia_final.tex"), (imp_sem, "importancia_sem_pontuacao.tex")):
        l = [r"\begin{tabular}{lrrr}", r"\toprule", r"Atributo & Queda AUC & Queda F1 & Peso conexão \\", r"\midrule"]
        for r in imp[:10]:
            l.append(f"{esc(r['atributo'])} & {f3(float(r['queda_auc']))} & {f3(float(r['queda_f1']))} & {f3(float(r['peso_conexao']))} \\\\")
        l += [r"\bottomrule", r"\end{tabular}"]
        escrever(nome, l)

    # --- erros da execução mediana
    l = [r"\begin{tabular}{lrrrrll}", r"\toprule", r"Jogador & G & MP & PTS & Prob. Good & Real & Previsto \\", r"\midrule"]
    for e in sorted(exp["final"]["mediana"]["erros"], key=lambda e: -float(limpo[e["jogador"]]["PTS"])):
        j = limpo[e["jogador"]]
        real = "Good" if e["real"] == 1 else "Bad"
        prev = "Good" if e["prob"] >= 0.5 else "Bad"
        l.append(f"{esc(e['jogador'])} & {int(float(j['G']))} & {f1(float(j['MP']))} & {f1(float(j['PTS']))} & {f3(e['prob'])} & {real} & {prev} \\\\")
    l += [r"\bottomrule", r"\end{tabular}"]
    escrever("erros.tex", l)


if __name__ == "__main__":
    print("gerando tabelas:")
    main()
