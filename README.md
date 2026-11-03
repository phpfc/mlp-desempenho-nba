# Previsão de desempenho de jogadores de basquete com MLP do zero

Trabalho da disciplina FBX4054 - Inteligência Computacional (UCS), semestre 2026/4.
Integrantes: Pedro Henrique Porto França Cornutti e Ramon Mielke.

## Entregáveis

- **Relatório:** [`relatorio/relatorio.pdf`](relatorio/relatorio.pdf), com fonte em `relatorio/relatorio.tex`
- **Apresentação:** [`apresentacao/apresentacao_resultados.pptx`](apresentacao/apresentacao_resultados.pptx), montada sobre o template do professor
- **Código:** [`src/`](src/), com os testes em [`tests/`](tests/)

A rede neural (`src/mlp.py`) e as métricas (`src/metricas.py`) são Python puro.
`pandas` é usado só na preparação dos dados e `matplotlib` só nas figuras.

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m unittest discover -s tests      # testes (gradiente numérico, XOR, métricas)
python src/preparacao_dados.py            # limpeza, divisão treino/teste, padronização
python src/experimentos.py                # validação cruzada, execuções finais, baselines (~4 min)
python src/importancia.py                 # importância das variáveis
python src/visualizacoes.py               # figuras em figuras/
python src/gerar_tabelas.py               # tabelas LaTeX em relatorio/tabelas/
cd relatorio && tectonic relatorio.tex    # gera relatorio/relatorio.pdf
python apresentacao/gerar_apresentacao.py # gera a apresentação a partir do template
```

## Estrutura

- `dados/nba_jogadores.csv` - dataset original; `dados/processados/` - dados limpos e padronizados
- `src/` - código-fonte; `tests/` - testes automatizados
- `resultados/` - `experimentos.json`, modelos treinados, importâncias, estatísticas
- `figuras/` - gráficos do relatório
- `relatorio/` - `relatorio.tex` + `relatorio.pdf` (com referências ABNT ligadas ao plano de ensino)
- `apresentacao/` - `apresentacao_resultados.pptx`, o template do professor e o script que gera a apresentação

O relatório é compilado com [Tectonic](https://tectonic-typesetting.github.io/), que baixa sozinho os pacotes LaTeX necessários.
