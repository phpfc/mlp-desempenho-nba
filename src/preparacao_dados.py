"""
Etapa 1 - Preparação dos dados.

Carrega o dataset de jogadores da NBA, faz a limpeza, codifica as variáveis
categóricas, separa treino/teste de forma estratificada e padroniza as
variáveis numéricas (z-score) usando apenas estatísticas do conjunto de treino.

Única biblioteca externa utilizada: pandas (manipulação de dados), conforme
permitido no enunciado. A divisão treino/teste e a padronização são feitas
manualmente.
"""
import json
import os
import random

import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_ENTRADA = os.path.join(RAIZ, "dados", "nba_jogadores.csv")
DIR_SAIDA = os.path.join(RAIZ, "dados", "processados")
DIR_RESULT = os.path.join(RAIZ, "resultados")

SEMENTE = 42
FRACAO_TESTE = 0.30
POSICOES = ["PG", "SG", "SF", "PF", "C"]


def carregar(caminho):
    df = pd.read_csv(caminho)
    print(f"Dataset bruto: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(df.head())
    print("\nValores faltantes por coluna:")
    print(df.isna().sum()[df.isna().sum() > 0])
    return df


def limpar(df, log):
    log["linhas_brutas"] = int(len(df))

    # 1) Coluna duplicada 'FG%.1' (mesmo valor de FG% com vírgula decimal, em texto)
    df = df.drop(columns=["FG%.1"])

    # 2) Linhas sem rótulo (inclui linha totalmente vazia)
    sem_alvo = df["Performance"].isna()
    log["linhas_sem_alvo"] = int(sem_alvo.sum())
    df = df[~sem_alvo].copy()

    # 3) Jogadores trocados de time aparecem com uma linha "TOT" (total da
    #    temporada) e uma linha por time. Mantemos apenas a linha TOT para não
    #    duplicar o mesmo jogador (o que vazaria informação entre treino/teste).
    jogadores_tot = set(df.loc[df["Tm"] == "TOT", "Player"])
    duplicada = df["Player"].isin(jogadores_tot) & (df["Tm"] != "TOT")
    log["jogadores_trocados"] = len(jogadores_tot)
    log["linhas_por_time_removidas"] = int(duplicada.sum())
    df = df[~duplicada].copy()

    # 4) Percentuais faltantes ocorrem quando o jogador não tentou nenhum
    #    arremesso daquele tipo (divisão 0/0). Preenchemos com 0.
    for col, tent in [("FG%", "FGA"), ("3P%", "3PA"), ("2P%", "2PA"), ("FT%", "FTA"), ("eFG%", "FGA")]:
        faltantes = df[col].isna()
        assert (df.loc[faltantes, tent] == 0).all(), f"{col} faltante com tentativas > 0"
        log[f"faltantes_{col}"] = int(faltantes.sum())
        df[col] = df[col].fillna(0.0)

    # 5) Posição: jogadores com posição dupla (ex.: 'PF-SF') ficam com a primeira
    df["Pos"] = df["Pos"].str.split("-").str[0]
    assert set(df["Pos"]).issubset(POSICOES), set(df["Pos"])

    # 6) Alvo binário
    df["alvo"] = (df["Performance"] == "Good").astype(int)

    log["linhas_limpas"] = int(len(df))
    log["classe_good"] = int(df["alvo"].sum())
    log["classe_bad"] = int((df["alvo"] == 0).sum())
    assert df.drop(columns=["Player", "Tm"]).isna().sum().sum() == 0
    return df


def codificar(df):
    """One-hot da posição; remove identificadores (nome e time)."""
    for p in POSICOES:
        df[f"Pos_{p}"] = (df["Pos"] == p).astype(int)
    atributos = [c for c in df.columns if c not in ("Player", "Tm", "Pos", "Performance", "alvo")]
    return df, atributos


def dividir_estratificado(df, fracao_teste, semente):
    """Separa treino/teste mantendo a proporção de classes (feito manualmente)."""
    rng = random.Random(semente)
    idx_teste = []
    for classe in (0, 1):
        indices = list(df.index[df["alvo"] == classe])
        rng.shuffle(indices)
        n_teste = round(len(indices) * fracao_teste)
        idx_teste.extend(indices[:n_teste])
    teste = df.loc[sorted(idx_teste)]
    treino = df.drop(index=idx_teste)
    return treino, teste


def padronizar(treino, teste, atributos):
    """z-score: (x - média) / desvio, com média e desvio calculados SÓ no treino."""
    parametros = {}
    treino_n, teste_n = treino.copy(), teste.copy()
    for col in atributos:
        if col.startswith("Pos_"):
            continue  # variáveis binárias não são padronizadas
        media = float(treino[col].mean())
        desvio = float(treino[col].std(ddof=0))
        if desvio == 0:
            desvio = 1.0
        parametros[col] = {"media": media, "desvio": desvio}
        treino_n[col] = (treino[col] - media) / desvio
        teste_n[col] = (teste[col] - media) / desvio
    return treino_n, teste_n, parametros


def estatisticas_descritivas(df, atributos):
    """Média por classe e correlação (ponto-bisserial = Pearson com alvo 0/1)."""
    linhas = []
    for col in atributos:
        if col.startswith("Pos_"):
            continue
        linhas.append({
            "atributo": col,
            "media_bad": df.loc[df["alvo"] == 0, col].mean(),
            "media_good": df.loc[df["alvo"] == 1, col].mean(),
            "desvio_geral": df[col].std(ddof=0),
            "correlacao_alvo": df[col].corr(df["alvo"]),
        })
    return pd.DataFrame(linhas).sort_values("correlacao_alvo", ascending=False)


def main():
    os.makedirs(DIR_SAIDA, exist_ok=True)
    os.makedirs(DIR_RESULT, exist_ok=True)
    log = {}

    df = carregar(ARQ_ENTRADA)
    df = limpar(df, log)
    df, atributos = codificar(df)
    print("\nAtributos de entrada:", atributos)

    treino, teste = dividir_estratificado(df, FRACAO_TESTE, SEMENTE)
    log["treino_total"] = int(len(treino))
    log["treino_good"] = int(treino["alvo"].sum())
    log["teste_total"] = int(len(teste))
    log["teste_good"] = int(teste["alvo"].sum())

    treino_n, teste_n, parametros = padronizar(treino, teste, atributos)

    colunas = ["Player"] + atributos + ["alvo"]
    df.to_csv(os.path.join(DIR_SAIDA, "dataset_limpo.csv"), index=False)
    treino_n[colunas].to_csv(os.path.join(DIR_SAIDA, "treino.csv"), index=False)
    teste_n[colunas].to_csv(os.path.join(DIR_SAIDA, "teste.csv"), index=False)
    with open(os.path.join(DIR_SAIDA, "padronizacao.json"), "w") as f:
        json.dump({"atributos": atributos, "parametros": parametros}, f, indent=2)
    with open(os.path.join(DIR_RESULT, "preparacao.json"), "w") as f:
        json.dump(log, f, indent=2)
    estatisticas_descritivas(df, atributos).to_csv(
        os.path.join(DIR_RESULT, "estatisticas_descritivas.csv"), index=False)

    print("\nResumo da preparação:")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print("\nCorrelação com o alvo (top 10):")
    print(estatisticas_descritivas(df, atributos).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
