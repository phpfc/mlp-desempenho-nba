"""
Monta a apresentação a partir do template do professor ("Análise dos Resultados").

Reaproveita os 7 slides do template (título, métricas, ROC, resultados, conclusões),
acrescenta slides com os layouts do próprio template e preenche tudo com os números
de resultados/experimentos.json e as figuras de figuras/.

Requer python-pptx (pip install python-pptx Pillow).
Uso: python apresentacao/gerar_apresentacao.py
"""
import json
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(RAIZ, "apresentacao", "template_analise_resultados.pptx")
SAIDA = os.path.join(RAIZ, "apresentacao", "apresentacao_resultados.pptx")
FIG = os.path.join(RAIZ, "figuras")

FONTE = "Tw Cen MT"                    # fonte usada em todos os slides do template
OLIVA = RGBColor(0x44, 0x4D, 0x26)     # dk2 do tema "Papel"
SALVIA = RGBColor(0x7C, 0x88, 0x6E)    # accent1 a 75% (cor dos títulos do template)
TINTA_CLARA = RGBColor(0xEE, 0xF1, 0xE9)
TINTA = RGBColor(0x1F, 0x1F, 0x1F)
TINTA2 = RGBColor(0x4A, 0x4A, 0x4A)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)

exp = json.load(open(os.path.join(RAIZ, "resultados", "experimentos.json")))


def br(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")


# --------------------------------------------------------------- utilitários
def runs_em(p, partes, tamanho=None, cor=None, negrito=None):
    """partes: str ou lista de (texto, {opções}) ."""
    if isinstance(partes, str):
        partes = [(partes, {})]
    for texto, op in partes:
        r = p.add_run()
        r.text = texto
        f = r.font
        f.name = FONTE
        tam = op.get("tamanho", tamanho)
        if tam:
            f.size = Pt(tam)
        c = op.get("cor", cor)
        if c is not None:
            f.color.rgb = c
        b = op.get("negrito", negrito)
        if b is not None:
            f.bold = b
        if op.get("italico"):
            f.italic = True


def sem_marcador(p):
    pPr = p._p.get_or_add_pPr()
    for filho in list(pPr):
        if filho.tag in (qn("a:buNone"), qn("a:buChar"), qn("a:buAutoNum")):
            pPr.remove(filho)
    pPr.insert(0, pPr.makeelement(qn("a:buNone"), {}))
    pPr.set("marL", "0")
    pPr.set("indent", "0")


def com_marcador(p):
    pPr = p._p.get_or_add_pPr()
    for filho in list(pPr):
        if filho.tag in (qn("a:buNone"), qn("a:buChar"), qn("a:buAutoNum"), qn("a:buFont"), qn("a:buClr")):
            pPr.remove(filho)
    pPr.set("marL", "228600")
    pPr.set("indent", "-228600")
    buClr = pPr.makeelement(qn("a:buClr"), {})
    clr = buClr.makeelement(qn("a:srgbClr"), {"val": "7C886E"})
    buClr.append(clr)
    buFont = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
    buChar = pPr.makeelement(qn("a:buChar"), {"char": "▪"})
    # ordem exigida pelo esquema: lnSpc, spcBef, spcAft, buClr, ..., buFont, buChar
    pPr.append(buClr)
    pPr.append(buFont)
    pPr.append(buChar)


def escrever(tf, paragrafos, tamanho=None, cor=None):
    """paragrafos: lista de dicts {partes, tamanho, cor, negrito, marcador, nivel, antes, alinhar}."""
    tf.clear()
    for i, d in enumerate(paragrafos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = d.get("nivel", 0)
        if "antes" in d:
            p.space_before = Pt(d["antes"])
        if d.get("marcador", False):
            com_marcador(p)
        else:
            sem_marcador(p)
        if "alinhar" in d:
            p.alignment = d["alinhar"]
        runs_em(p, d["partes"], d.get("tamanho", tamanho), d.get("cor", cor), d.get("negrito"))


def caixa(slide, x, y, w, h, paragrafos, tamanho=14, cor=TINTA, ancora=MSO_ANCHOR.TOP, nome=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if nome:
        tb.name = nome
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancora
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    escrever(tf, paragrafos, tamanho, cor)
    return tb


def forma(slide, tipo, x, y, w, h, preenchimento, nome=None):
    s = slide.shapes.add_shape(tipo, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = preenchimento
    s.line.fill.background()
    s.shadow.inherit = False
    if nome:
        s.name = nome
    if tipo == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.06
    return s


def circulo_numero(slide, x, y, d, n):
    c = forma(slide, MSO_SHAPE.OVAL, x, y, d, d, OLIVA, nome=f"Número {n}")
    tf = c.text_frame
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    escrever(tf, [{"partes": str(n), "negrito": True, "alinhar": PP_ALIGN.CENTER}], tamanho=16, cor=BRANCO)


def figura(slide, arquivo, x, y, w=None, h=None, descricao=""):
    caminho = os.path.join(FIG, arquivo)
    largura, altura = Image.open(caminho).size
    if w is not None and h is None:
        h = w * altura / largura
    elif h is not None and w is None:
        w = h * largura / altura
    pic = slide.shapes.add_picture(caminho, Inches(x), Inches(y), Inches(w), Inches(h))
    pic.name = arquivo
    pic._element.nvPicPr.cNvPr.set("descr", descricao)
    return pic, w, h


def tabela(slide, linhas, x, y, larguras, altura_linha, tamanho=14, cabecalho=True,
           negrito_linhas=(), primeira_coluna_destaque=False, alinhar_num=True, nome=None):
    nl, nc = len(linhas), len(linhas[0])
    gf = slide.shapes.add_table(nl, nc, Inches(x), Inches(y), Inches(sum(larguras)), Inches(altura_linha * nl))
    if nome:
        gf.name = nome
    t = gf.table
    t.first_row = cabecalho
    t.horz_banding = False
    for c, wdt in enumerate(larguras):
        t.columns[c].width = Inches(wdt)
    for r in range(nl):
        t.rows[r].height = Inches(altura_linha)
        for c in range(nc):
            cel = t.cell(r, c)
            cel.margin_left = cel.margin_right = Inches(0.07)
            cel.margin_top = cel.margin_bottom = Inches(0.02)
            cel.vertical_anchor = MSO_ANCHOR.MIDDLE
            eh_cab = cabecalho and r == 0
            if eh_cab:
                fundo, cor, neg = OLIVA, BRANCO, True
            elif primeira_coluna_destaque and c == 0:
                fundo, cor, neg = TINTA_CLARA, OLIVA, True
            else:
                fundo = TINTA_CLARA if (r % 2 == 0 and cabecalho) else BRANCO
                cor, neg = TINTA, (r in negrito_linhas)
            cel.fill.solid()
            cel.fill.fore_color.rgb = fundo
            alinhar = PP_ALIGN.LEFT if (c == 0 or not alinhar_num) else PP_ALIGN.CENTER
            escrever(cel.text_frame, [{"partes": str(linhas[r][c]), "negrito": neg, "alinhar": alinhar}],
                     tamanho=tamanho, cor=cor)
    return gf


def remover(shape):
    el = shape._element
    el.getparent().remove(el)


def placeholder(slide, idx):
    for ph in slide.placeholders:
        if ph.placeholder_format.idx == idx:
            return ph
    return None


def titulo(slide, texto):
    escrever(slide.shapes.title.text_frame, [{"partes": texto}])


def notas(slide, texto):
    tf = slide.notes_slide.notes_text_frame
    if tf is not None:
        tf.text = texto


# ------------------------------------------------------------------ montagem
prs = Presentation(TEMPLATE)
layouts = {l.name: l for l in prs.slide_layouts}
L_CONTEUDO = layouts["Título e conteúdo"]
L_LEGENDA = layouts["Conteúdo com legenda"]
t1, t2, t3, t4, t5, t6, t7 = list(prs.slides)

fin = exp["final"]
res = fin["resumo"]
med = fin["mediana"]
semB = exp["cenario_sem_pontuacao"]["resumo"]
regra = exp["baselines"]["regra_pts"]
maj = exp["baselines"]["majoritaria"]
ativ = {r["parametros"]["ativacao"]: r["resumo"] for r in exp["ativacoes"]}
melhor_grid = next(r for r in exp["grid"] if r["nome"] == exp["melhor_grid"])
ep = exp["analise_epocas"]
f1_ep = {t["epoca"]: t["f1_val"] for t in ep["tabela"]}

# ---- slide 1 (template): título e identificação
sub = placeholder(t1, 1)
sub.top, sub.height = Inches(3.98), Inches(1.6)
escrever(sub.text_frame, [
    {"partes": [("MLP do zero para prever o desempenho de jogadores da NBA", {"negrito": True, "cor": OLIVA})]},
    {"partes": "Disciplina: FBX4054 - Inteligência Computacional"},
    {"partes": "Alunos: Pedro Henrique Porto França Cornutti e Ramon Mielke"},
    {"partes": "Semestre: 2026/4"},
])
notas(t1, "Apresentação do projeto da Atividade Avaliativa 2: uma rede MLP implementada do zero, em Python puro, "
          "para classificar jogadores da NBA em desempenho Good ou Bad.")

# ---- dados e preparação
s_dados = prs.slides.add_slide(L_CONTEUDO)
titulo(s_dados, "Dados e preparação")
remover(placeholder(s_dados, 1))
passos = [
    ("624 linhas e 31 colunas", "Removidas 2 linhas sem rótulo e a coluna FG%.1, cópia de FG% em texto"),
    ("57 jogadores trocados de time", "Mantida só a linha TOT de cada um: −114 linhas, sem o mesmo jogador no treino e no teste"),
    ("Percentuais vazios", "Aparecem quando não há tentativas (0/0) e foram preenchidos com 0"),
    ("508 jogadores e 31 atributos", "Posição em one-hot, z-score com média e desvio do treino e divisão estratificada 70/30"),
]
for i, (cab, desc) in enumerate(passos):
    y = 1.95 + i * 1.16
    forma(s_dados, MSO_SHAPE.ROUNDED_RECTANGLE, 0.9, y, 6.6, 1.02, TINTA_CLARA, nome=f"Etapa {i + 1}")
    circulo_numero(s_dados, 1.08, y + 0.24, 0.54, i + 1)
    caixa(s_dados, 1.85, y + 0.12, 5.45, 0.34, [{"partes": cab, "negrito": True}], tamanho=17, cor=OLIVA)
    caixa(s_dados, 1.85, y + 0.48, 5.45, 0.5, [{"partes": desc}], tamanho=14, cor=TINTA2)
figura(s_dados, "fig01_classes.png", 7.85, 1.95, w=4.55, descricao="Barras com 321 Bad e 34 Good no treino e 138 Bad e 15 Good no teste")
caixa(s_dados, 7.85, 4.95, 4.55, 0.62, [{"partes": "9,6%", "negrito": True}], tamanho=40, cor=OLIVA)
caixa(s_dados, 7.85, 5.62, 4.55, 0.8, [{"partes": "dos jogadores são Good: prever sempre Bad já acerta 90%, então a acurácia sozinha engana"}],
      tamanho=14, cor=TINTA2)
notas(s_dados, "Limpeza com pandas: linhas sem rótulo, coluna duplicada e jogadores trocados de time (mantivemos o total TOT "
               "para não ter o mesmo jogador em treino e teste). Percentuais vazios vêm de divisões 0/0. Padronização z-score "
               "com estatísticas só do treino e divisão estratificada 70/30, feitas à mão. Classes muito desbalanceadas.")

# ---- o rótulo é um limiar de pontos
s_rotulo = prs.slides.add_slide(L_LEGENDA)
titulo(s_rotulo, "O rótulo é quase um limiar de pontos")
remover(placeholder(s_rotulo, 1))
escrever(placeholder(s_rotulo, 2).text_frame, [
    {"partes": "Esse achado orienta a análise: a regra simples PTS ≥ 20,2 vira a referência que a MLP precisa alcançar."}])
figura(s_rotulo, "fig02_pts_por_classe.png", 0.55, 0.7, w=6.9, descricao="Pontos por jogo de cada jogador por classe; todos os Good acima de 20 pontos")
stats = [("20,5", "pontos por jogo é o mínimo entre os 49 jogadores Good"),
         ("1", "jogador Bad passa de 20 pontos (Terry Rozier, 21,4)"),
         ("0,74", "é a correlação de PTS com o alvo; FT, FTA e FG passam de 0,7")]
for i, (num, txt) in enumerate(stats):
    x = 0.55 + i * 2.35
    caixa(s_rotulo, x, 4.0, 2.2, 0.62, [{"partes": num, "negrito": True}], tamanho=34, cor=OLIVA)
    caixa(s_rotulo, x, 4.66, 2.2, 0.95, [{"partes": txt}], tamanho=14, cor=TINTA2)
caixa(s_rotulo, 0.55, 5.9, 6.9, 0.7, [{"partes": "Aproveitamento (FG%, 2P%, eFG%) tem correlação abaixo de 0,08: ser Good é produzir volume, não ser eficiente."}],
      tamanho=14, cor=TINTA)
notas(s_rotulo, "Na análise exploratória descobrimos que o rótulo Good corresponde, na prática, a marcar 20 ou mais pontos por jogo. "
                "Isso explica a maior parte dos resultados e dos erros da rede.")

# ---- arquitetura
s_arq = prs.slides.add_slide(L_CONTEUDO)
titulo(s_arq, "Arquitetura da MLP")
remover(placeholder(s_arq, 1))
_, _, h_arq = figura(s_arq, "fig04_arquitetura.png", 0.9, 1.95, w=6.5, descricao="Diagrama da rede 31-16-1")
caixa(s_arq, 0.9, 1.95 + h_arq + 0.15, 6.5, 0.6,
      [{"partes": "Totalmente conectada; a saída é a probabilidade de Good e o limiar de decisão é 0,5."}], tamanho=14, cor=TINTA2)
cfg = exp["config_final"]
tabela(s_arq, [
    ("Entradas", "31 atributos padronizados"),
    ("Camada oculta", f"{cfg['camadas'][0]} neurônios, sigmoide"),
    ("Saída", "1 neurônio, sigmoide"),
    ("Parâmetros", "529 pesos e vieses"),
    ("Perda", "entropia cruzada binária"),
    ("Otimização", f"mini-lotes de {exp['lote']}, η = {br(cfg['taxa'], 1)}"),
    ("Épocas", str(cfg["epocas"])),
    ("Inicialização", "Glorot uniforme"),
    ("Execuções", f"{len(fin['execucoes'])} sementes diferentes"),
], 7.75, 1.95, [1.75, 2.95], 0.44, tamanho=14, cabecalho=False, primeira_coluna_destaque=True, alinhar_num=False,
    nome="Tabela da arquitetura")
notas(s_arq, "Arquitetura final escolhida por validação cruzada: 31 entradas, 16 neurônios ocultos e 1 saída, todos com sigmoide. "
             "Avaliamos 5 execuções com sementes de inicialização diferentes.")

# ---- implementação
s_impl = prs.slides.add_slide(L_CONTEUDO)
titulo(s_impl, "Implementação do zero em Python puro")
remover(placeholder(s_impl, 1))
cards = [
    ("Propagação direta", "Cada camada calcula z = Σ w·a + b e a = σ(z). A saída sigmoide é a probabilidade de Good."),
    ("Retropropagação", "O erro na saída é δ = ŷ − y (sigmoide com entropia cruzada) e volta camada a camada pela regra da cadeia."),
    ("Descida do gradiente", "Mini-lotes de 32 exemplos, embaralhados a cada época, com w ← w − η·∂L/∂w."),
]
for i, (cab, txt) in enumerate(cards):
    x = 0.9 + i * 3.925
    forma(s_impl, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.95, 3.7, 2.3, TINTA_CLARA, nome=f"Cartão {cab}")
    circulo_numero(s_impl, x + 0.25, 2.2, 0.54, i + 1)
    caixa(s_impl, x + 0.95, 2.22, 2.6, 0.5, [{"partes": cab, "negrito": True}], tamanho=18, cor=OLIVA, ancora=MSO_ANCHOR.MIDDLE)
    caixa(s_impl, x + 0.25, 2.95, 3.2, 1.25, [{"partes": txt}], tamanho=16, cor=TINTA)
forma(s_impl, MSO_SHAPE.ROUNDED_RECTANGLE, 0.9, 4.5, 11.55, 1.95, OLIVA, nome="Faixa de verificação")
caixa(s_impl, 1.2, 4.7, 11.0, 0.4, [{"partes": "Verificação: só listas, math e random, sem NumPy nem scikit-learn", "negrito": True}],
      tamanho=17, cor=BRANCO)
verif = ["Gradiente analítico igual às diferenças finitas: erro relativo < 10⁻⁵ com sigmoide, tanh e ReLU",
         "Problema XOR, que não é linearmente separável, aprendido com 100% de acerto",
         "8 testes automatizados aprovados, incluindo métricas e curva ROC"]
for i, txt in enumerate(verif):
    caixa(s_impl, 1.2 + i * 3.75, 5.25, 3.45, 1.05, [{"partes": txt}], tamanho=15, cor=BRANCO)
notas(s_impl, "A rede, o treino e as métricas usam só Python puro. Conferimos a retropropagação comparando o gradiente com "
              "diferenças finitas em todos os parâmetros e treinando o XOR.")

# ---- ajuste de hiperparâmetros
s_cv = prs.slides.add_slide(L_LEGENDA)
titulo(s_cv, "Ajuste por validação cruzada")
remover(placeholder(s_cv, 1))
mg = melhor_grid["resumo"]["f1"]
escrever(placeholder(s_cv, 2).text_frame, [
    {"partes": f"{len(exp['grid'])} combinações × {exp['folds']} folds = {len(exp['grid']) * exp['folds']} treinos, só no conjunto de treino."},
    {"partes": f"Melhor: 16 neurônios, η = 0,1, sem ponderação. F1 de validação {br(mg['media'])} ± {br(mg['desvio'])}.", "antes": 10},
])
_, _, h_cv = figura(s_cv, "fig05_grid_cv.png", 0.55, 0.7, w=6.9, descricao="F1 médio de validação por arquitetura, taxa de aprendizado e ponderação")
achados = [("Taxa de aprendizado decide: ", "com η = 0,01 a rede não converge em 300 épocas e a 16-8 fica presa na classe Bad."),
           ("Arquitetura pouco importa: ", "8, 16, 32 ou 16-8 neurônios empatam quando η é adequado."),
           ("Ponderar a classe Good (w+ = 9,4) ", "aumenta o recall, mas reduz a precisão e não melhora o F1.")]
caixa(s_cv, 0.55, 0.7 + h_cv + 0.45, 6.9, 3.0,
      [{"partes": [(a, {"negrito": True, "cor": OLIVA}), (b, {})], "antes": 0 if i == 0 else 16} for i, (a, b) in enumerate(achados)],
      tamanho=17, cor=TINTA)
notas(s_cv, "Grid de 24 combinações avaliado com validação cruzada estratificada de 5 folds, sem usar o teste. "
            "O critério foi o F1 da classe Good, porque a acurácia é pouco informativa com 90% de Bad.")

# ---- épocas e ativação
s_ep = prs.slides.add_slide(L_CONTEUDO)
titulo(s_ep, "Épocas e função de ativação")
remover(placeholder(s_ep, 1))
_, _, h_ep = figura(s_ep, "fig06_curvas_treinamento.png", 0.9, 1.95, w=7.4, descricao="Curvas de perda e de F1 e acurácia de validação por época")
caixa(s_ep, 8.65, 1.95, 3.8, 0.95, [{"partes": str(ep["epoca_escolhida"]), "negrito": True}], tamanho=54, cor=OLIVA)
caixa(s_ep, 8.65, 2.95, 3.8, 0.8, [{"partes": f"épocas maximizam o F1 de validação ({br(f1_ep[ep['epoca_escolhida']])})"}], tamanho=16, cor=TINTA)
caixa(s_ep, 8.65, 3.85, 3.8, 1.2, [{"partes": f"Com 300 épocas o F1 cai para {br(f1_ep[300])}: a rede ganha confiança nos casos fáceis, não nos de fronteira."}],
      tamanho=14, cor=TINTA2)
linhas_at = [("Ativação", "F1 médio (5 folds)", "AUC")]
for nome in ("sigmoide", "tanh", "relu"):
    a = ativ[nome]
    linhas_at.append((nome if nome != "relu" else "ReLU", f"{br(a['f1']['media'])} ± {br(a['f1']['desvio'])}", br(a["auc"]["media"])))
tabela(s_ep, linhas_at, 0.9, 1.95 + h_ep + 0.25, [2.2, 3.2, 2.0], 0.33, tamanho=14, negrito_linhas=(1,), nome="Tabela de ativações")
notas(s_ep, "As curvas médias dos 5 folds mostram o F1 de validação máximo em 110 épocas. A sigmoide ficou um pouco acima de tanh e ReLU, "
            "mas a diferença está dentro da variação entre folds.")

# ---- slide 2 do template: resultados por execução
titulo(t2, "Resultados por execução no teste")
remover(placeholder(t2, 1))
cab = ("Execução", "VP", "FP", "VN", "FN", "Acurácia", "Erro", "Recall", "Especif.", "Precisão", "F1", "AUC")
chaves = ("acuracia", "erro", "recall", "especificidade", "precisao", "f1", "auc")
linhas_ex = [cab]
for e in fin["execucoes"]:
    m = e["metricas"]
    rot = f"Semente {e['semente']}" + (" (mediana)" if e["semente"] == med["semente"] else "")
    linhas_ex.append((rot, m["VP"], m["FP"], m["VN"], m["FN"], *[br(m[k]) for k in chaves]))
linhas_ex.append(("Média", "", "", "", "", *[br(res[k]["media"]) for k in chaves]))
linhas_ex.append(("Desvio padrão", "", "", "", "", *[br(res[k]["desvio"]) for k in chaves]))
tabela(t2, linhas_ex, 0.9, 1.95, [2.05, 0.52, 0.52, 0.58, 0.52, 1.05, 0.9, 0.9, 1.0, 1.05, 0.9, 0.9], 0.42, tamanho=14,
       negrito_linhas=(len(linhas_ex) - 2, len(linhas_ex) - 1), nome="Tabela por execução")
caixa(t2, 0.9, 5.55, 11.55, 0.9, [
    {"partes": f"{exp['n_teste']} jogadores de teste (15 Good e 138 Bad), treino com os {exp['n_treino']} jogadores de treino. "
               "Entre as execuções muda só a semente de inicialização. Recall = sensitividade."}], tamanho=14, cor=TINTA2)
notas(t2, "Cinco execuções da configuração final no conjunto de teste, usado uma única vez. Os resultados são estáveis: "
          "a rede encontra 14 ou 15 dos 15 Good e comete de 2 a 4 falsos positivos.")

# ---- matriz de confusão e ROC
s_roc = prs.slides.add_slide(L_CONTEUDO)
titulo(s_roc, "Matriz de confusão e curva ROC")
remover(placeholder(s_roc, 1))
_, w_mc, _ = figura(s_roc, "fig07_matriz_confusao.png", 0.9, 1.9, h=3.55, descricao="Matriz de confusão: 135 VN, 3 FP, 1 FN, 14 VP")
figura(s_roc, "fig08_roc.png", 0.9 + w_mc + 0.25, 1.9, h=3.55, descricao="Curva ROC da execução mediana com AUC 0,995")
mm = med["metricas"]
caixa(s_roc, 9.25, 1.95, 3.2, 3.6, [
    {"partes": f"Execução mediana (semente {med['semente']})", "negrito": True, "tamanho": 16, "cor": OLIVA},
    {"partes": f"{mm['VP']} de {mm['VP'] + mm['FN']} jogadores Good encontrados", "marcador": True, "antes": 8},
    {"partes": f"{mm['FP']} falsos positivos entre {mm['FP'] + mm['VN']} Bad", "marcador": True, "antes": 8},
    {"partes": f"AUC {br(mm['auc'])}: com 3 falsos positivos já se acham todos os Good", "marcador": True, "antes": 8},
], tamanho=15, cor=TINTA)
erros = sorted(med["erros"], key=lambda e: -e["prob"])
caixa(s_roc, 0.9, 5.65, 11.55, 0.8, [
    {"partes": [("Os 4 erros estão entre 19,5 e 21,4 pontos por jogo, ", {"negrito": True, "cor": OLIVA}),
                ("na fronteira do rótulo: Terry Rozier, Dejounte Murray, Cade Cunningham e Tyrese Maxey.", {})]}],
    tamanho=14, cor=TINTA)
notas(s_roc, "Matriz e ROC da execução mediana. A AUC de 0,995 mostra que a rede ordena quase perfeitamente os jogadores. "
             "Os erros são todos jogadores com cerca de 20 pontos por jogo, onde o próprio rótulo é ambíguo.")

# ---- slide 6 do template: tabela comparativa
titulo(t6, "Resultados: tabela comparativa")
remover(placeholder(t6, 1))
rB = regra["metricas"]
comp = [("Modelo", "Acurácia", "Erro", "Recall", "Especif.", "Precisão", "F1", "AUC"),
        ("Classe majoritária (sempre Bad)", *[br(maj[k]) for k in chaves]),
        (f"Regra PTS ≥ {br(regra['limiar_pts'], 1)}", *[br(rB[k]) for k in chaves]),
        ("MLP sem pontuação (22 atributos)", *[br(semB[k]["media"]) for k in chaves]),
        ("MLP completa (31 atributos)", *[br(res[k]["media"]) for k in chaves])]
tabela(t6, comp, 0.9, 1.95, [3.99, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08, 1.08], 0.48, tamanho=15, negrito_linhas=(4,),
       nome="Tabela comparativa")
forma(t6, MSO_SHAPE.ROUNDED_RECTANGLE, 0.9, 4.65, 5.65, 1.55, TINTA_CLARA, nome="Melhor configuração")
caixa(t6, 1.15, 4.85, 5.15, 1.2, [
    {"partes": "Melhor configuração da MLP", "negrito": True, "cor": OLIVA, "tamanho": 17},
    {"partes": f"31-16-1, sigmoide, η = 0,1, {cfg['epocas']} épocas e sem ponderação: F1 {br(res['f1']['media'])} e AUC "
               f"{br(res['auc']['media'])} no teste (média de 5 execuções).", "antes": 6}], tamanho=15, cor=TINTA)
forma(t6, MSO_SHAPE.ROUNDED_RECTANGLE, 6.8, 4.65, 5.65, 1.55, OLIVA, nome="Regra de referência")
caixa(t6, 7.05, 4.85, 5.15, 1.2, [
    {"partes": "A regra de um atributo vence", "negrito": True, "tamanho": 17},
    {"partes": f"Com F1 {br(rB['f1'])}, o limiar PTS ≥ {br(regra['limiar_pts'], 1)} supera a rede: o rótulo é praticamente esse limiar.",
     "antes": 6}], tamanho=15, cor=BRANCO)
notas(t6, "Comparação no teste. A MLP supera de longe a classe majoritária, mas a regra de um único atributo é ainda melhor, "
          "porque o rótulo foi definido praticamente por pontos por jogo. Sem os atributos de pontuação a rede ainda chega a AUC 0,981.")

# ---- importância das variáveis
s_imp = prs.slides.add_slide(L_LEGENDA)
titulo(s_imp, "Variáveis com maior impacto")
remover(placeholder(s_imp, 1))
escrever(placeholder(s_imp, 2).text_frame, [
    {"partes": "Importância por permutação: embaralhar um atributo no teste e medir a queda de AUC."},
    {"partes": f"Sem os 9 atributos de pontuação, a rede ainda atinge AUC {br(semB['auc']['media'])}.", "antes": 10}])
_, _, h_imp = figura(s_imp, "fig09_importancia.png", 0.55, 0.7, w=6.9, descricao="Queda de AUC por atributo nos dois cenários")
caixa(s_imp, 0.55, 0.7 + h_imp + 0.45, 6.9, 2.6, [
    {"partes": [("Modelo completo: ", {"negrito": True, "cor": OLIVA}),
                ("PTS, 2P, FT, FTA e FG. São redundantes (PTS = 2·2P + 3·3P + FT), por isso cada um derruba pouco a AUC.", {})]},
    {"partes": [("Sem pontuação: ", {"negrito": True, "cor": OLIVA}),
                ("erros de posse (TOV) e minutos (MP) dominam. Quem pontua muito fica mais tempo com a bola.", {})], "antes": 16},
], tamanho=17, cor=TINTA)
notas(s_imp, "A importância por permutação aponta PTS e seus componentes. Sem eles, a rede usa o perfil de uso do jogador, "
             "principalmente erros de posse e minutos por jogo.")

# ---- slide 7 do template: conclusões
corpo = placeholder(t7, 1)
escrever(corpo.text_frame, [
    {"partes": "Concluir", "negrito": True, "cor": OLIVA, "tamanho": 21},
    {"partes": f"A MLP do zero funciona: acurácia {br(res['acuracia']['media'] * 100, 1)}%, recall {br(res['recall']['media'] * 100, 1)}% "
               f"e AUC {br(res['auc']['media'])}, estável entre 5 execuções", "marcador": True, "antes": 8},
    {"partes": "Aprendizagens", "negrito": True, "cor": OLIVA, "tamanho": 21, "antes": 22},
    {"partes": "Taxa de aprendizado e épocas importam mais que a arquitetura", "marcador": True, "antes": 8},
    {"partes": "Com classes desbalanceadas, olhar recall, precisão, F1 e ROC", "marcador": True, "antes": 8},
    {"partes": "Entender o rótulo é essencial: uma regra de um atributo supera a rede", "marcador": True, "antes": 8},
    {"partes": "Trabalhos futuros", "negrito": True, "cor": OLIVA, "tamanho": 21, "antes": 22},
    {"partes": "Ajustar o limiar pela ROC, parada antecipada e otimizador Adam", "marcador": True, "antes": 8},
    {"partes": "Remover atributos redundantes e usar um rótulo de desempenho mais rico", "marcador": True, "antes": 8},
], tamanho=18, cor=TINTA)
escrever(placeholder(t7, 2).text_frame, [
    {"partes": f"Análise adicional: sem os atributos de pontuação, a rede ainda atinge AUC {br(semB['auc']['media'])} usando erros de posse e minutos."}])
notas(t7, "Fechamento: a implementação está correta e estável, os ajustes que mais importaram foram taxa de aprendizado e épocas, "
          "e o principal aprendizado é que entender o rótulo muda a leitura dos resultados.")

# ---- referências
s_ref = prs.slides.add_slide(L_CONTEUDO)
titulo(s_ref, "Referências")
refs = [
    ("ADAMI, A. G. ", "Métodos de avaliação de desempenho", ". Slides da disciplina FBX4054. Caxias do Sul: UCS, 2026."),
    ("ADAMI, A. G.; BOFF, E. ", "Redes neurais artificiais", ". Slides da disciplina FBX4054. Caxias do Sul: UCS, 2026."),
    ("FACELI, K. et al. ", "Inteligência artificial", ": uma abordagem de aprendizado de máquina. 2. ed. Rio de Janeiro: LTC, 2021."),
    ("FAWCETT, T. An introduction to ROC analysis. ", "Pattern Recognition Letters", ", v. 27, n. 8, p. 861–874, 2006."),
    ("HAYKIN, S. ", "Redes neurais", ": princípios e prática. 2. ed. Porto Alegre: Bookman, 2001."),
    ("LUGER, G. F. ", "Inteligência artificial", ". 6. ed. São Paulo: Pearson, 2013."),
    ("RUMELHART, D. E.; HINTON, G. E.; WILLIAMS, R. J. Learning representations by back-propagating errors. ", "Nature",
     ", v. 323, p. 533–536, 1986."),
    ("RUSSELL, S. J.; NORVIG, P. ", "Inteligência artificial", ": uma abordagem moderna. 4. ed. Rio de Janeiro: GEN LTC, 2022."),
]
escrever(placeholder(s_ref, 1).text_frame,
         [{"partes": [(a, {}), (b, {"negrito": True}), (c, {})], "antes": 0 if i == 0 else 7} for i, (a, b, c) in enumerate(refs)]
         + [{"partes": [("Lista completa, com 20 referências, no relatório.", {"italico": True, "cor": TINTA2})], "antes": 12}],
         tamanho=14, cor=TINTA)
notas(s_ref, "Principais referências: a bibliografia básica do plano de ensino, os slides da disciplina e as fontes clássicas da MLP.")

# ------------------------------------------------------------- ordem final
ordem = [t1, s_dados, s_rotulo, s_arq, s_impl, s_cv, s_ep, t3, t4, t5, t2, s_roc, t6, s_imp, t7, s_ref]
lst = prs.slides._sldIdLst
ids = [s.slide_id for s in ordem]
elementos = {int(e.get("id")): e for e in lst}
assert sorted(ids) == sorted(elementos), "ordem não cobre todos os slides"
for e in list(lst):
    lst.remove(e)
for i in ids:
    lst.append(elementos[i])

prs.core_properties.title = "Análise dos Resultados - MLP para jogadores da NBA"
prs.core_properties.author = "Pedro Henrique Porto França Cornutti; Ramon Mielke"
prs.save(SAIDA)
print("salvo:", SAIDA, "-", len(ordem), "slides")
