# Roteiro – Vídeo de apresentação (10 min)

Dica: grave a tela com OBS Studio ou com o próprio Google Meet, mostrando os slides, o notebook e o artigo. Publique no YouTube como **não listado** e cole o link no README.

| Tempo | Parte | O que mostrar / falar |
|---|---|---|
| 0:00–0:45 | **Abertura** | Nome(s), disciplina SIN 393, professor. Título do projeto. "Classificar automaticamente 6 condições de folhas de arroz a partir de fotos." |
| 0:45–1:45 | **Motivação** | Arroz é alimento básico; doenças derrubam a produtividade; o diagnóstico visual depende de especialista. Visão computacional pode ajudar o produtor. |
| 1:45–2:45 | **Trabalhos relacionados** | 2 ou 3 destaques: Ahad et al. (CNNs + transfer learning, +17 p.p.), Mehnaz & Islam (ResNet50 superou MaxViT), modelos híbridos ViT-CNN. Por isso a pergunta: CNN ou Transformer? |
| 2:45–4:00 | **Dataset** | Link do Kaggle, 6 classes, figura de amostras. **Remoção de duplicatas por MD5 antes da divisão** (explicar o vazamento de dados). Divisão estratificada 70/15/15, semente 42 (mostrar a tabela). |
| 4:00–5:30 | **Métodos** | ResNet-50 (convolução, conexões residuais) x ViT-B/16 (patches 16×16 + atenção). Transfer learning do ImageNet. Data augmentation: mostrar cada transformação e **por que** a matiz quase não varia (cor = sintoma). Hiperparâmetros iguais; early stopping pela validação; teste usado uma única vez. |
| 5:30–6:00 | **Código** | Mostrar rapidamente o repositório e o notebook no Colab rodando. |
| 6:00–8:30 | **Resultados** | Tabela de resultados (macro x weighted e por que estão próximas). Responder Q1 (qual arquitetura venceu e o custo do ViT) e Q2 (efeito do DA nas curvas). Matriz de confusão: o par mais confundido. Análise qualitativa: 2 ou 3 erros e as causas prováveis. |
| 8:30–9:30 | **Conclusão e limitações** | Resposta curta para Q1 e Q2. Limitações: uma única divisão/semente; só duplicatas exatas. Trabalhos futuros: validação cruzada, Grad-CAM, modelos leves. |
| 9:30–10:00 | **Encerramento** | Link do GitHub, agradecimento. |

Se o grupo tiver 3 pessoas, uma sugestão de divisão: (1) abertura → dataset; (2) métodos → código; (3) resultados → encerramento.
