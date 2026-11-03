"""
Etapa 4 - Métricas de avaliação implementadas manualmente (sem bibliotecas).

Classe positiva = 1 (Good), classe negativa = 0 (Bad).
"""


def matriz_confusao(y_real, y_pred):
    vp = fp = vn = fn = 0
    for r, p in zip(y_real, y_pred):
        if r == 1 and p == 1:
            vp += 1
        elif r == 0 and p == 1:
            fp += 1
        elif r == 0 and p == 0:
            vn += 1
        else:
            fn += 1
    return {"VP": vp, "FP": fp, "VN": vn, "FN": fn}


def _div(a, b):
    return a / b if b else 0.0


def calcular_metricas(y_real, y_pred):
    mc = matriz_confusao(y_real, y_pred)
    vp, fp, vn, fn = mc["VP"], mc["FP"], mc["VN"], mc["FN"]
    acuracia = _div(vp + vn, vp + fp + vn + fn)
    precisao = _div(vp, vp + fp)
    recall = _div(vp, vp + fn)                 # sensibilidade / taxa de VP
    especificidade = _div(vn, vn + fp)         # taxa de VN
    f1 = _div(2 * precisao * recall, precisao + recall)
    return {
        "acuracia": acuracia,
        "erro": 1.0 - acuracia,
        "precisao": precisao,
        "recall": recall,
        "especificidade": especificidade,
        "f1": f1,
        **mc,
    }


def curva_roc(y_real, pontuacoes):
    """Curva ROC: varre todos os limiares possíveis (as pontuações ordenadas de
    forma decrescente) e calcula (FPR, TPR) em cada um. Retorna os pontos e a
    área sob a curva (AUC), calculada pela regra do trapézio."""
    positivos = sum(1 for r in y_real if r == 1)
    negativos = len(y_real) - positivos
    ordem = sorted(range(len(y_real)), key=lambda i: -pontuacoes[i])
    pontos = [(0.0, 0.0, float("inf"))]
    vp = fp = 0
    i = 0
    while i < len(ordem):
        limiar = pontuacoes[ordem[i]]
        # empates de pontuação são processados juntos
        while i < len(ordem) and pontuacoes[ordem[i]] == limiar:
            if y_real[ordem[i]] == 1:
                vp += 1
            else:
                fp += 1
            i += 1
        pontos.append((_div(fp, negativos), _div(vp, positivos), limiar))
    auc = 0.0
    for (x0, y0, _), (x1, y1, _) in zip(pontos, pontos[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return pontos, auc


def media(valores):
    return sum(valores) / len(valores) if valores else 0.0


def desvio_padrao(valores):
    if len(valores) < 2:
        return 0.0
    m = media(valores)
    return (sum((v - m) ** 2 for v in valores) / (len(valores) - 1)) ** 0.5
