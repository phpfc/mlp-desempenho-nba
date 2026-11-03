"""Testes de corretude da MLP e das métricas (python -m unittest discover tests)."""
import copy
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from mlp import MLP  # noqa: E402
from metricas import calcular_metricas, curva_roc  # noqa: E402


class TesteGradiente(unittest.TestCase):
    """Compara o gradiente da retropropagação com diferenças finitas."""

    def _verificar(self, ativacao, peso_positivo):
        rng = random.Random(1)
        rede = MLP(4, [5, 3], ativacao=ativacao, semente=7)
        # vieses iniciam em zero; perturbá-los evita que um pré-ativação caia
        # exatamente na quina da ReLU (z = 0), onde a diferença finita não vale
        for b in rede.vieses:
            for j in range(len(b)):
                b[j] = rng.uniform(-0.1, 0.1)
        X = [[rng.uniform(-1, 1) for _ in range(4)] for _ in range(6)]
        y = [1, 0, 1, 1, 0, 0]
        gW, gb = rede.gradientes(X, y, peso_positivo)
        eps = 1e-6
        maior_erro = 0.0
        for l in range(len(rede.pesos)):
            for j in range(len(rede.pesos[l])):
                for i in range(len(rede.pesos[l][j])):
                    original = rede.pesos[l][j][i]
                    rede.pesos[l][j][i] = original + eps
                    mais = rede.perda(X, y, peso_positivo)
                    rede.pesos[l][j][i] = original - eps
                    menos = rede.perda(X, y, peso_positivo)
                    rede.pesos[l][j][i] = original
                    numerico = (mais - menos) / (2 * eps)
                    erro = abs(numerico - gW[l][j][i]) / max(1e-8, abs(numerico) + abs(gW[l][j][i]))
                    maior_erro = max(maior_erro, erro)
                original = rede.vieses[l][j]
                rede.vieses[l][j] = original + eps
                mais = rede.perda(X, y, peso_positivo)
                rede.vieses[l][j] = original - eps
                menos = rede.perda(X, y, peso_positivo)
                rede.vieses[l][j] = original
                numerico = (mais - menos) / (2 * eps)
                erro = abs(numerico - gb[l][j]) / max(1e-8, abs(numerico) + abs(gb[l][j]))
                maior_erro = max(maior_erro, erro)
        self.assertLess(maior_erro, 1e-5, f"{ativacao}: erro relativo {maior_erro}")

    def test_sigmoide(self):
        self._verificar("sigmoide", 1.0)

    def test_tanh_ponderado(self):
        self._verificar("tanh", 5.0)

    def test_relu(self):
        self._verificar("relu", 1.0)


class TesteAprendizado(unittest.TestCase):
    def test_xor(self):
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [0, 1, 1, 0]
        rede = MLP(2, [8], ativacao="tanh", semente=3)
        hist = rede.treinar(X, y, epocas=1500, taxa=0.5, tamanho_lote=None)
        self.assertEqual(rede.prever(X), y)
        self.assertLess(hist["perda_treino"][-1], hist["perda_treino"][0])

    def test_exportar_importar(self):
        rede = MLP(3, [4], semente=1)
        copia = MLP.importar(copy.deepcopy(rede.exportar()))
        x = [0.1, -0.2, 0.3]
        self.assertAlmostEqual(rede.propagar(x)[-1][0], copia.propagar(x)[-1][0])


class TesteMetricas(unittest.TestCase):
    def test_matriz_e_metricas(self):
        y_real = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        m = calcular_metricas(y_real, y_pred)
        self.assertEqual((m["VP"], m["FP"], m["VN"], m["FN"]), (2, 1, 2, 1))
        self.assertAlmostEqual(m["acuracia"], 4 / 6)
        self.assertAlmostEqual(m["precisao"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["especificidade"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 2 / 3)

    def test_sem_positivos_previstos(self):
        m = calcular_metricas([1, 0, 1], [0, 0, 0])
        self.assertEqual(m["precisao"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)
        self.assertEqual(m["especificidade"], 1.0)

    def test_roc_auc(self):
        _, auc = curva_roc([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1])
        self.assertAlmostEqual(auc, 1.0)
        _, auc = curva_roc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
        self.assertAlmostEqual(auc, 0.75)
        pontos, _ = curva_roc([1, 0], [0.5, 0.5])
        self.assertEqual(pontos[-1][:2], (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
