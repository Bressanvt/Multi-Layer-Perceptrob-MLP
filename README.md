# MLPClassifier — Multi-Layer Perceptron com PyTorch

Implementação didática de uma **MLP (Multi-Layer Perceptron)** para classificação binária, construída do zero com PyTorch.

## Projetos

### Fatec-MLP (`app.ipynb`)

MLP completa com suporte a:
- Múltiplas camadas ocultas configuráveis
- Ativações: ReLU, Tanh
- Otimizadores: Adam, SGD
- Regularização: Dropout, L2
- Early stopping
- Mini-batch training
- Plot da fronteira de decisão (2D)
- Plot do histórico de treinamento (loss + acurácia)

**Dependências:** `numpy`, `torch`, `matplotlib`

**Datasets:**
| Arquivo | Features | Amostras | Descrição |
|---------|----------|----------|-----------|
| `train_dataset1.csv` | x1, x2 | 140 | Linearmente separável |
| `test_dataset1.csv` | x1, x2 | 60 | Teste para dataset1 |
| `train_dataset2.csv` | x1, x2 | 176 | **Rosquinha (círculos concêntricos)** — não linear |
| `test_dataset2.csv` | x1, x2 | 76 | Teste para dataset2 |

### Aula 17 (`MLP.ipynb`)

Implementação clássica do **Perceptron de uma camada** (single-layer) com NumPy puro — sem frameworks de ML.

Testado em 3 cenários:
- **Dataset 1 (2D):** Linearmente separável — Perceptron converge (~98% acurácia)
- **Dataset 2 (2D):** **Não linear** — Perceptron falha (~50%, chute aleatório)
- **Dataset 3 (10 features):** Parcialmente linear (~90% acurácia)
- **Dataset multiclasse (50 features):** 4 classes (0, 1, 2, 3)

**Dependências:** `numpy`, `matplotlib`

**Principais conceitos demonstrados:**
- Limitação do Perceptron: só aprende fronteiras lineares
- Necessidade de camadas ocultas para problemas não lineares
- A rosquinha (dataset2) como exemplo clássico de não linearidade

## Licença

Projeto educacional — Fatec.
