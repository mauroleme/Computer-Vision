# Passo 4 — Avaliação Final no Split Test
**Visão por Computador · MEEC / IPB 2025–2026**
Caio Sant'Ana Oliveira (52963) · Mauro da Silva Leme (a52965)

---

## Visão geral

O Passo 4 é a **avaliação honesta e definitiva** do classificador. Usa o split TEST — dados que nunca foram vistos em nenhuma fase anterior (nem calibração, nem validação). Corre **uma única vez**, no final do projecto.

> ⚠️ Regra fundamental: não ajustar limiares depois de ver estes resultados. Fazê-lo seria data leakage — os números deixariam de representar performance real em dados novos.

---

## Ficheiros do Passo 4

| Ficheiro | Tipo | O que contém |
|---|---|---|
| `passo4_avaliacao_final.py` | Script | Avaliação completa no split test |
| `passo4_test_predicoes.csv` | Resultado | Predição de cada ROI do split test |
| `passo4_confusion.png` | Imagem | Matriz de confusão final |
| `passo4_acuracias.png` | Imagem | Accuracy global e por parque |
| `passo4_erros_dist.png` | Imagem | Distribuição de accuracy por imagem |
| `passo4_erros_piores.png` | Imagem | Amostra das piores imagens por parque |

---

## Porquê três splits (train / valid / test)?

| Split | Usado para | Visto pelo classificador? |
|---|---|---|
| **train** | Calibrar os limiares (passo3) | Sim — os limiares foram optimizados aqui |
| **valid** | Verificar limiares durante calibração | Sim — indirectamente (escolha de lógica 2/4 vs 3/4) |
| **test** | Avaliação final (passo4) | **Nunca** — até ao momento da avaliação |

Se o passo4 avaliasse no train ou valid, os números seriam optimistas — o classificador teria sido ajustado para esses dados. O test dá uma estimativa realista do que acontece com imagens reais nunca vistas.

---

## O que o script avalia

- **Accuracy global** — percentagem de vagas correctamente classificadas em todos os parques
- **Accuracy por parque** — G28, G40 e G100 separados
- **Precision, Recall e F1** — por parque e por classe (Livre/Ocupada)
- **Distribuição de accuracy por imagem** — mostra se o classificador é consistente ou tem imagens onde falha muito
- **Piores imagens** — identifica os casos onde o erro é maior (para análise qualitativa: são imagens com sol intenso? Chuva? Carros brancos?)

---

## O ficheiro `passo4_test_predicoes.csv`

Tem uma linha por ROI classificada no split test:

| Coluna | Descrição |
|---|---|
| `split` | sempre "test" |
| `filename` | nome do ficheiro de imagem |
| `parque` | G28, G40 ou G100 |
| `ann_id` | ID da anotação COCO |
| `label_gt` | ground truth: "livre" ou "ocupada" |
| `label_pred` | predição do classificador |
| `correcto` | 1 se acertou, 0 se errou |
| `f1` a `f4` | valores das 4 features para essa ROI |

Este CSV permite análise post-hoc: identificar padrões nos erros, comparar por condição meteorológica, por hora do dia, etc.

---

## Resultados finais (split test)

| Parque | Accuracy | Observação |
|---|---|---|
| G28 | ~92% | Melhor resultado — ROIs grandes, features robustas |
| G40 | ~91% | Segundo melhor — dark_ratio ajuda com carros escuros |
| G100 | ~89% | Menor — ROIs muito pequenas (~23×46 px) dificultam GLCM |
| **Global** | **~91%** | Média ponderada pelos três parques |

Estes valores são consistentes com os do passo3 no valid — não há queda significativa no test, o que indica ausência de overfitting.

---

## Ligação ao pipeline

```
passo3_limiares.json  →  passo4_avaliacao_final.py  →  passo4_*.csv / passo4_*.png
```

O passo4 não gera nenhum ficheiro que o passo5 ou passo6 precisem. É um passo terminal de avaliação — os seus outputs são para o relatório e apresentação, não para uso computacional posterior.
