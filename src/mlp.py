"""
Etapa 2 - Rede neural MLP (perceptron multicamadas) implementada do zero.

Apenas Python puro: listas, `math` e `random`. Nenhuma biblioteca numérica ou de
aprendizado de máquina é utilizada.

Convenções:
  - pesos[l][j][i] : peso que liga a entrada i ao neurônio j da camada l
  - vieses[l][j]   : viés do neurônio j da camada l
  - A camada de saída tem um único neurônio com ativação sigmoide, que produz a
    probabilidade estimada de a instância pertencer à classe positiva (Good).
  - Função de perda: entropia cruzada binária (BCE), opcionalmente ponderada
    para a classe positiva (peso_positivo) de modo a compensar o desbalanceamento.
"""
import math
import random


# ----------------------------------------------------------------------------
# Funções de ativação e suas derivadas (expressas em função da própria ativação)
# ----------------------------------------------------------------------------
def sigmoide(z):
    # forma numericamente estável
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def deriv_sigmoide(a):
    return a * (1.0 - a)


def tanh(z):
    return math.tanh(z)


def deriv_tanh(a):
    return 1.0 - a * a


def relu(z):
    return z if z > 0.0 else 0.0


def deriv_relu(a):
    return 1.0 if a > 0.0 else 0.0


ATIVACOES = {
    "sigmoide": (sigmoide, deriv_sigmoide),
    "tanh": (tanh, deriv_tanh),
    "relu": (relu, deriv_relu),
}

EPS = 1e-12  # evita log(0)


class MLP:
    """MLP para classificação binária treinada por retropropagação + descida do gradiente."""

    def __init__(self, n_entradas, camadas_ocultas=(16,), ativacao="sigmoide", semente=0):
        if ativacao not in ATIVACOES:
            raise ValueError(f"ativação desconhecida: {ativacao}")
        self.n_entradas = n_entradas
        self.camadas_ocultas = list(camadas_ocultas)
        self.ativacao = ativacao
        self.semente = semente
        self.f, self.df = ATIVACOES[ativacao]
        self.rng = random.Random(semente)
        self.tamanhos = [n_entradas] + self.camadas_ocultas + [1]
        self.pesos = []
        self.vieses = []
        self._inicializar_pesos()

    # ------------------------------------------------------------------ pesos
    def _inicializar_pesos(self):
        """Inicialização de Glorot (sigmoide/tanh) ou He (relu), uniforme.
        Vieses iniciam em zero."""
        n_camadas = len(self.tamanhos) - 1
        for l in range(n_camadas):
            n_in, n_out = self.tamanhos[l], self.tamanhos[l + 1]
            oculta = l < n_camadas - 1
            if self.ativacao == "relu" and oculta:
                limite = math.sqrt(6.0 / n_in)
            else:
                limite = math.sqrt(6.0 / (n_in + n_out))
            W = [[self.rng.uniform(-limite, limite) for _ in range(n_in)] for _ in range(n_out)]
            b = [0.0] * n_out
            self.pesos.append(W)
            self.vieses.append(b)

    def numero_parametros(self):
        return sum(len(W) * len(W[0]) + len(b) for W, b in zip(self.pesos, self.vieses))

    # ---------------------------------------------------------------- forward
    def propagar(self, x):
        """Propagação direta. Retorna a lista de ativações de todas as camadas
        (a primeira é a própria entrada e a última é a saída da rede)."""
        ativacoes = [x]
        a = x
        ultima = len(self.pesos) - 1
        for l, (W, b) in enumerate(zip(self.pesos, self.vieses)):
            f = sigmoide if l == ultima else self.f
            a = [f(sum(w * ai for w, ai in zip(wj, a)) + bj) for wj, bj in zip(W, b)]
            ativacoes.append(a)
        return ativacoes

    def prever_proba(self, X):
        return [self.propagar(x)[-1][0] for x in X]

    def prever(self, X, limiar=0.5):
        return [1 if p >= limiar else 0 for p in self.prever_proba(X)]

    # ------------------------------------------------------------------ perda
    def perda(self, X, y, peso_positivo=1.0):
        """Entropia cruzada binária média (ponderada para a classe positiva)."""
        total = 0.0
        for x, yi in zip(X, y):
            p = self.propagar(x)[-1][0]
            if yi == 1:
                total -= peso_positivo * math.log(p + EPS)
            else:
                total -= math.log(1.0 - p + EPS)
        return total / len(X)

    # --------------------------------------------------------------- backprop
    def _retropropagar(self, ativacoes, y, peso, acum_W, acum_b):
        """Retropropagação do erro para UMA instância; soma os gradientes nos
        acumuladores (acum_W, acum_b).

        Para a saída sigmoide com perda BCE, o gradiente da perda em relação
        ao potencial de ativação do neurônio de saída é  peso * (y_prev - y).
        """
        # delta da camada de saída
        delta = [peso * (ativacoes[-1][0] - y)]
        for l in range(len(self.pesos) - 1, -1, -1):
            a_ant = ativacoes[l]          # ativações da camada anterior (entradas desta camada)
            gW, gb = acum_W[l], acum_b[l]
            for j, dj in enumerate(delta):
                gb[j] += dj
                linha = gW[j]
                for i, ai in enumerate(a_ant):
                    linha[i] += dj * ai
            if l > 0:
                # propaga o erro para a camada anterior:
                # delta_ant[i] = f'(a_ant[i]) * sum_j W[l][j][i] * delta[j]
                W = self.pesos[l]
                delta = [
                    self.df(a_ant[i]) * sum(W[j][i] * delta[j] for j in range(len(delta)))
                    for i in range(len(a_ant))
                ]

    def _acumuladores_zerados(self):
        acum_W = [[[0.0] * len(W[0]) for _ in W] for W in self.pesos]
        acum_b = [[0.0] * len(b) for b in self.vieses]
        return acum_W, acum_b

    def gradientes(self, X, y, peso_positivo=1.0):
        """Gradiente médio da perda em relação a todos os pesos e vieses
        (usado no treino e na verificação numérica dos testes)."""
        acum_W, acum_b = self._acumuladores_zerados()
        for x, yi in zip(X, y):
            peso = peso_positivo if yi == 1 else 1.0
            self._retropropagar(self.propagar(x), yi, peso, acum_W, acum_b)
        m = float(len(X))
        for gW, gb in zip(acum_W, acum_b):
            for j in range(len(gW)):
                gb[j] /= m
                linha = gW[j]
                for i in range(len(linha)):
                    linha[i] /= m
        return acum_W, acum_b

    def _atualizar_pesos(self, grad_W, grad_b, taxa, l2):
        """Passo da descida do gradiente: w <- w - taxa * (dL/dw + l2 * w)."""
        for W, b, gW, gb in zip(self.pesos, self.vieses, grad_W, grad_b):
            for j in range(len(W)):
                b[j] -= taxa * gb[j]
                linha, glinha = W[j], gW[j]
                for i in range(len(linha)):
                    linha[i] -= taxa * (glinha[i] + l2 * linha[i])

    # ------------------------------------------------------------------ treino
    def treinar(self, X, y, epocas=100, taxa=0.1, tamanho_lote=32, peso_positivo=1.0,
                l2=0.0, X_val=None, y_val=None, avaliar_a_cada=1, verboso=False):
        """Descida do gradiente em mini-lotes (tamanho_lote=None => lote completo).

        Retorna o histórico: época, perda de treino e, se houver validação,
        perda e métricas de validação.
        """
        from metricas import calcular_metricas  # import local evita dependência circular

        n = len(X)
        lote = n if tamanho_lote is None else tamanho_lote
        indices = list(range(n))
        historico = {"epoca": [], "perda_treino": [], "perda_val": [], "f1_val": [], "acuracia_val": []}

        for epoca in range(1, epocas + 1):
            self.rng.shuffle(indices)
            for inicio in range(0, n, lote):
                ids = indices[inicio:inicio + lote]
                grad_W, grad_b = self.gradientes([X[i] for i in ids], [y[i] for i in ids], peso_positivo)
                self._atualizar_pesos(grad_W, grad_b, taxa, l2)

            if epoca % avaliar_a_cada == 0 or epoca == epocas:
                historico["epoca"].append(epoca)
                historico["perda_treino"].append(self.perda(X, y, peso_positivo))
                if X_val is not None:
                    historico["perda_val"].append(self.perda(X_val, y_val, peso_positivo))
                    m = calcular_metricas(y_val, self.prever(X_val))
                    historico["f1_val"].append(m["f1"])
                    historico["acuracia_val"].append(m["acuracia"])
                if verboso:
                    msg = f"época {epoca:4d}  perda_treino={historico['perda_treino'][-1]:.4f}"
                    if X_val is not None:
                        msg += f"  perda_val={historico['perda_val'][-1]:.4f}  f1_val={historico['f1_val'][-1]:.3f}"
                    print(msg)
        return historico

    # ------------------------------------------------------------ persistência
    def exportar(self):
        return {
            "n_entradas": self.n_entradas,
            "camadas_ocultas": self.camadas_ocultas,
            "ativacao": self.ativacao,
            "semente": self.semente,
            "pesos": self.pesos,
            "vieses": self.vieses,
        }

    @classmethod
    def importar(cls, d):
        rede = cls(d["n_entradas"], d["camadas_ocultas"], d["ativacao"], d["semente"])
        rede.pesos = [[linha[:] for linha in W] for W in d["pesos"]]
        rede.vieses = [b[:] for b in d["vieses"]]
        return rede
