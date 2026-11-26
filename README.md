# Classificação de Doenças em Folhas de Arroz com CNN e Vision Transformer

Projeto da disciplina **SIN 393 – Introdução à Visão Computacional (2026)**, Universidade Federal de Viçosa – Campus Rio Paranaíba, Prof. João Fernando Mari.

**Autores:** Vinícius Matoso Dias 8785, Paulo Roberto 8158 e Gabriel Rodrigues 8125

**Vídeo de apresentação (YouTube):** *https://youtu.be/1PLxgVUjOAA*

---

## 1. Descrição

O projeto classifica imagens de folhas de arroz em **6 classes**: *Bacterial Leaf Blight*, *Brown Spot*, *Healthy Rice Leaf*, *Leaf Blast*, *Leaf Scald* e *Sheath Blight*, usando modelos de Deep Learning do `torchvision` em PyTorch.

**Dataset:** [Rice Disease Dataset – Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)

### Questões experimentais
1. **CNN x Transformer:** ResNet-50 x ViT-B/16, ambos com *transfer learning* (pesos do ImageNet) e *fine-tuning* completo.
2. **Impacto do data augmentation:** cada modelo é treinado com e sem *data augmentation*.

Isso resulta em 4 experimentos com a mesma divisão de dados, os mesmos hiperparâmetros e a mesma semente:

| Experimento | Modelo | Data augmentation |
|---|---|---|
| `resnet50_aug` | ResNet-50 | Sim |
| `resnet50_noaug` | ResNet-50 | Não |
| `vit_b_16_aug` | ViT-B/16 | Sim |
| `vit_b_16_noaug` | ViT-B/16 | Não |

### Metodologia
- **Achado importante sobre o dataset:** a pasta `Rice_Leaf_AUG` do Kaggle mistura as fotos originais com **cópias já aumentadas** (`aug_0_*.jpg`, 224×224, cerca de 77% dos arquivos). Se fossem mantidas, versões da mesma foto cairiam no treino e no teste (**vazamento de dados**), e o experimento sobre data augmentation perderia o sentido. Por isso, **usamos apenas as fotos originais** (flag `--keep-aug-files` para desativar).
- **Limpeza:** verificação de arquivos corrompidos, remoção de **duplicatas exatas (MD5)** e **agrupamento de quase-duplicatas** com hash perceptual (dHash nas 8 orientações de rotação de 90° e espelhamento). A divisão é feita **por grupo**, então fotos quase iguais nunca ficam em conjuntos diferentes.
- **Cache:** as imagens são redimensionadas uma vez (lado maior = 512 px) para acelerar o treino.
- **Divisão hold-out estratificada (por grupo)** 70% / 15% / 15% (treino / validação / teste) com `sklearn.model_selection.train_test_split`, `stratify=label` e `random_state=42`. As listas ficam salvas em `results/split_*.csv`.
- **Pré-processamento:** redimensionamento para 224×224 e normalização com as estatísticas do ImageNet.
- **Data augmentation (só no treino):**
  - `RandomResizedCrop` (escala 0,7–1,0): simula variação de distância e enquadramento sem cortar a lesão.
  - Flips horizontal/vertical e rotação de ±20°: a orientação da folha é arbitrária.
  - `ColorJitter` leve (brilho/contraste/saturação 0,2; matiz 0,02): simula iluminação de campo. A matiz quase não varia porque a cor é sintoma da doença.
- **Treino:** AdamW (lr = 1e-4, weight decay = 1e-4), cosine annealing, batch 32, até 20 épocas, *early stopping* com paciência 5 pelo **F1-macro da validação**, precisão mista (AMP).
- **Avaliação (somente no teste, uma única vez):** matriz de confusão, acurácia, precisão, recall e F1, com agregação **macro** e **weighted**, além de métricas por classe.
- **Análise qualitativa:** exemplos de acertos e dos erros de maior confiança, pares de classes mais confundidos e erros em comum entre os modelos.

## 2. Estrutura do repositório

```
├── main.py                      # roda todos os experimentos e gera tabelas e figuras
├── src/
│   ├── data.py                  # índice, limpeza, divisão estratificada, transforms
│   ├── models.py                # ResNet-50 / ViT-B/16 (torchvision) com nova cabeça
│   ├── engine.py                # laços de treino/validação, early stopping, predição
│   └── reports.py               # métricas, figuras e tabelas LaTeX
├── notebooks/
│   └── SIN393_rice_disease.ipynb  # notebook pronto para o Google Colab
├── paper/
│   ├── main.tex, refs.bib       # artigo no template IEEE Conference
│   ├── generated/               # tabelas e números gerados por main.py
│   └── figures/                 # figuras geradas por main.py
├── results/                     # métricas, históricos, predições (gerado)
├── docs/roteiro_video.md        # roteiro da apresentação de 10 min
└── requirements.txt
```

## 3. Como executar

### Opção A: Google Colab (recomendado, precisa de GPU)
1. Abra `notebooks/SIN393_rice_disease.ipynb` no Colab (*Arquivo → Abrir notebook → GitHub*).
2. *Ambiente de execução → Alterar tipo → GPU T4*.
3. Execute as células em ordem. O dataset é baixado com `kagglehub`. Se for pedido login, gere um token em *kaggle.com → Settings → API*.
4. Ao final, baixe `resultados_sin393.zip`, extraia na raiz do repositório e faça commit de `results/` e `paper/`.

### Opção B: Localmente
**Requisitos:** Python ≥ 3.10 e, de preferência, uma GPU NVIDIA com CUDA (em CPU funciona, mas é muito lento).

```bash
git clone https://github.com/matosod1as/sin393-rice-disease-classification.git
cd sin393-rice-disease-classification
pip install -r requirements.txt

# baixar o dataset
python -c "import kagglehub; print(kagglehub.dataset_download('anshulm257/rice-disease-dataset'))"

# rodar os 4 experimentos (use o caminho impresso acima)
python main.py --data-dir <CAMINHO_DO_DATASET> --epochs 20 --batch-size 32
```

Parâmetros úteis: `--models resnet50 vit_b_16` (também aceita `efficientnet_b0`, `convnext_tiny`, `swin_t`, `resnet18`), `--aug yes no`, `--save-weights`, `--max-per-class 20 --epochs 1` (teste rápido).

### Saídas
- `results/summary.csv`: métricas de teste de todos os experimentos
- `results/<experimento>/`: `history.csv`, `metrics.json`, `test_predictions.csv`
- `results/error_analysis.json`: pares mais confundidos e erros em comum
- `paper/figures/*.png` e `paper/generated/*.tex`: usados diretamente pelo artigo

### Compilar o artigo
Envie a pasta `paper/` para o Overleaf (ou rode `pdflatex main && bibtex main && pdflatex main && pdflatex main`). Os números do texto vêm de `paper/generated/macros.tex`. Os trechos em **vermelho** são interpretações que devem ser conferidas com os resultados reais.

## 4. Resultados

Conjunto de teste: 136 imagens (fotos originais, sem as cópias `aug_*`). GPU Tesla T4 no Google Colab.

| Modelo | DA | Acurácia | F1-macro | F1-weighted | Erros | Tempo de treino |
|---|---|---|---|---|---|---|
| ResNet-50 | Sim | 91,91% | 89,49% | 91,95% | 11 | 2,2 min |
| ResNet-50 | Não | 88,97% | 84,48% | 88,96% | 15 | 1,2 min |
| **ViT-B/16** | **Sim** | **94,85%** | **91,46%** | **94,81%** | **7** | 2,7 min |
| ViT-B/16 | Não | 91,91% | 87,45% | 91,80% | 11 | 1,6 min |

**Principais conclusões**
- **Q1 (CNN x Transformer):** o ViT-B/16 superou a ResNet-50 nas duas condições (+2 a +3 p.p. de F1-macro), mas tem 3,6× mais parâmetros, e a diferença corresponde a poucas imagens.
- **Q2 (data augmentation):** melhorou as duas arquiteturas (+5,0 p.p. na ResNet-50 e +4,0 p.p. no ViT). Sem DA, os modelos chegam a 100% de acurácia no treino por volta da época 6 (sobreajuste).
- A classe mais difícil é *Leaf scald* (F1 de 73,7%; é a classe com menos fotos), confundida principalmente com *Brown Spot*.

Figuras: `paper/figures/` (matrizes de confusão, curvas, análise qualitativa).

## 5. Referências principais
- He et al. *Deep Residual Learning for Image Recognition*. CVPR, 2016.
- Dosovitskiy et al. *An Image is Worth 16x16 Words*. ICLR, 2021.
- Mehnaz & Islam. *Rice Leaf Disease Detection: A Comparative Study Between CNN, Transformer and Non-neural Network Architectures*. arXiv:2501.06740, 2025.
- Ahad et al. *Comparison of CNN-based deep learning architectures for rice diseases classification*. Artificial Intelligence in Agriculture, 2023.
