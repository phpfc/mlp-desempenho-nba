"""
Etapa 5 - Importância das variáveis, calculada manualmente para o modelo final
(execução mediana) e para o cenário sem atributos de pontuação.

1) Importância por permutação: embaralha os valores de UM atributo no conjunto
   de teste (quebrando sua relação com o alvo) e mede a queda de AUC e de F1.
   Média de R repetições com sementes diferentes.
2) Pesos de conexão (generalização do método de Garson): produto das matrizes
   de pesos em valor absoluto, |W_1|^T |W_2|^T ... |W_saida|^T, normalizado.
"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp import MLP  # noqa: E402
from metricas import calcular_metricas, curva_roc, media  # noqa: E402
from experimentos import ler_csv, ATRIBUTOS_VOLUME, DIR_DADOS, DIR_RESULT  # noqa: E402

REPETICOES = 20


def importancia_permutacao(rede, X, y, colunas, repeticoes=REPETICOES, semente=0):
    rng = random.Random(semente)
    probas = rede.prever_proba(X)
    _, auc_base = curva_roc(y, probas)
    f1_base = calcular_metricas(y, [1 if p >= 0.5 else 0 for p in probas])["f1"]
    resultado = []
    for j, col in enumerate(colunas):
        quedas_auc, quedas_f1 = [], []
        for _ in range(repeticoes):
            valores = [x[j] for x in X]
            rng.shuffle(valores)
            X_perm = [x[:j] + [v] + x[j + 1:] for x, v in zip(X, valores)]
            p = rede.prever_proba(X_perm)
            _, auc = curva_roc(y, p)
            f1 = calcular_metricas(y, [1 if q >= 0.5 else 0 for q in p])["f1"]
            quedas_auc.append(auc_base - auc)
            quedas_f1.append(f1_base - f1)
        resultado.append({"atributo": col, "queda_auc": media(quedas_auc), "queda_f1": media(quedas_f1)})
    return resultado, auc_base, f1_base


def pesos_conexao(rede):
    """Vetor de importância por atributo de entrada via produto de |pesos|."""
    # começa pela camada de saída: vetor |w_saida| (tamanho = neurônios da última oculta)
    vetor = [abs(w) for w in rede.pesos[-1][0]]
    for W in reversed(rede.pesos[:-1]):
        # W[j][i]: entrada i -> neurônio j ; nova importância da entrada i = sum_j |W[j][i]| * vetor[j]
        vetor = [sum(abs(W[j][i]) * vetor[j] for j in range(len(W))) for i in range(len(W[0]))]
    total = sum(vetor)
    return [v / total for v in vetor]


def calcular(nome_modelo, excluir, arq_saida):
    with open(os.path.join(DIR_RESULT, nome_modelo)) as f:
        rede = MLP.importar(json.load(f))
    X, y, colunas, _ = ler_csv(os.path.join(DIR_DADOS, "teste.csv"), excluir=excluir)
    assert len(colunas) == rede.n_entradas
    perm, auc_base, f1_base = importancia_permutacao(rede, X, y, colunas)
    conexao = pesos_conexao(rede)
    for r, c in zip(perm, conexao):
        r["peso_conexao"] = c
    perm.sort(key=lambda r: -r["queda_auc"])
    with open(os.path.join(DIR_RESULT, arq_saida), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["atributo", "queda_auc", "queda_f1", "peso_conexao"])
        w.writeheader()
        w.writerows(perm)
    print(f"\n{nome_modelo}: AUC base={auc_base:.4f}  F1 base={f1_base:.4f}")
    for r in perm[:12]:
        print(f"  {r['atributo']:<7} queda_auc={r['queda_auc']:+.4f} queda_f1={r['queda_f1']:+.4f} conexao={r['peso_conexao']:.3f}")


if __name__ == "__main__":
    calcular("modelo_final.json", (), "importancia_final.csv")
    calcular("modelo_sem_pontuacao.json", ATRIBUTOS_VOLUME, "importancia_sem_pontuacao.csv")
