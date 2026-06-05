# Passo 3 — Calibração dos Classificadores por Parque
**Visão por Computador · MEEC / IPB 2025–2026**
Caio Sant'Ana Oliveira (52963) · Mauro da Silva Leme (a52965)

---

## Visão geral

O Passo 3 é o coração da calibração. Recebe as features eleitas pelo Passo 2b e determina, para cada parque, **qual o valor de corte (threshold) óptimo** de cada feature e **como combinar os votos** das 4 features para chegar à decisão final (Livre ou Ocupada).

Não usa redes neuronais nem aprendizagem automática no sentido tradicional. "Treinar" aqui significa: varrer sistematicamente possíveis limiares e escolher o que maximiza a performance.

---

## Ficheiros do Passo 3

| Ficheiro | Tipo | O que contém |
|---|---|---|
| `passo3_classificador.py` | Script | Código de calibração e avaliação |
| `passo3_limiares.json` | Resultado principal | Limiares e lógica de votação por parque |
| `passo3_valid_predicoes.csv` | Resultado | Predição de cada vaga do split valid (143.316 ROIs) |
| `passo3_confusion.png` | Imagem | Matriz de confusão global |
| `passo3_acuracias.png` | Imagem | Accuracy global e por parque |
| `passo3_calib_summary.png` | Imagem | Resumo da calibração por parque |

---

## Fluxo interno do script (5 fases)

```
FASE 1 — Extracção de features de amostra do train
         3.000 ROIs por parque (1.500 livres + 1.500 ocupadas)
              ↓
FASE 2 — Sweep de limiares por feature
         Para cada feature, testar 300 valores → escolher o melhor threshold
              ↓
FASE 3 — Comparar lógicas de votação
         2-em-3 (sem Harris) vs 3-em-4 (com Harris) → guardar a melhor
              ↓
FASE 4 — Validar no split VALID completo
         Todas as ROIs do valid (não apenas amostra)
              ↓
FASE 5 — Guardar resultados
         passo3_limiares.json + CSV + plots
```

---

## FASE 1 — Amostragem do split de treino

- **3.000 ROIs por parque** = 1.500 livres + 1.500 ocupadas
- Amostradas aleatoriamente do split train (semente fixa = 42, para reprodutibilidade)
- As imagens são carregadas uma vez e todas as ROIs seleccionadas dessa imagem são processadas em bloco — optimização que evita recarregar o mesmo ficheiro várias vezes

---

## FASE 2 — Sweep de limiares

Para cada feature (F1, F2, F3, F4) e cada parque:

1. Calcular o valor da feature em todas as 3.000 ROIs de calibração
2. Definir 300 pontos de teste entre o mínimo e o máximo observados
3. Para cada ponto, testar **duas direcções**:
   - `gt` (greater than): classifica como OC se `valor > threshold`
   - `lt` (less than): classifica como OC se `valor < threshold`
4. Calcular a accuracy em ambas as direcções
5. Guardar o par (threshold, direcção) que dá maior accuracy

**A direcção não é assumida à partida** — é descoberta automaticamente. Por exemplo:
- GLCM Homogeneidade → detecta-se que `lt` é a direcção correcta (homogeneidade baixa = veículo)
- Sobel, Prewitt, Harris → detectam-se como `gt` (gradiente/cantos altos = veículo)

---

## FASE 3 — Lógica de votação

Cada feature dá um voto: **Livre** ou **Ocupada**. A decisão final é por maioria.

Foram comparadas duas lógicas:

| Lógica | Regra | Vantagem | Desvantagem |
|---|---|---|---|
| **2-em-3** | OC se ≥ 2 de {F1, F2, F3} votam OC | Mais permissivo — menos falsos negativos | Mais falsos positivos (sombras, reflexos) |
| **3-em-4** | OC se ≥ 3 de {F1, F2, F3, F4} votam OC | Mais robusto | Pode perder carros claros (fracos em todas as features) |

**Regra de decisão para incluir F4 (Harris):**
O Harris só é incluído (lógica 3-em-4) se a accuracy com ele for pelo menos **0.1 pp superior** à versão sem ele. Se não melhorar, o sistema usa apenas F1+F2+F3 com maioria simples.

Nos três parques, a lógica 3-em-4 foi seleccionada.

---

## FASE 4 — Validação completa

Após calibrar os limiares, o classificador é avaliado no **split valid inteiro** — não na amostra de calibração. Esta separação é fundamental: garante que os números reportados não são inflados por avaliar nos mesmos dados usados para calibrar.

O resultado fica em `passo3_valid_predicoes.csv` com 143.316 linhas (uma por ROI do valid):

| Coluna | Descrição |
|---|---|
| `split` | sempre "valid" |
| `filename` | nome do ficheiro de imagem |
| `parque` | G28, G40 ou G100 |
| `ann_id` | ID da anotação COCO |
| `label_gt` | ground truth: "livre" ou "ocupada" |
| `label_pred` | predição do classificador |
| `correcto` | 1 se acertou, 0 se errou |
| `f1` a `f4` | valores das 4 features para essa ROI |

---

## Resultados: `passo3_limiares.json`

Este ficheiro é o **modelo treinado** — é o único que os passos seguintes (4, 5, 6) precisam. Contém para cada parque:

### G28 (28 vagas por imagem)

| Feature | Descritor | Pipeline | Threshold | Direcção | Acc individual |
|---|---|---|---|---|---|
| F1 | std_intensity | unsharp | 0.2014 | gt | 91.8% |
| F2 | glcm_contrast | gauss | 5.4450 | gt | 91.9% |
| F3 | std_intensity | raw | 0.1584 | gt | 91.1% |
| F4 | n_cantos (t=0.05, d=3) | unsharp | 5.137 | gt | 84.5% |
| **Combinado 3/4** | votação maioritária | — | — | — | **92.3%** |

### G40 (40 vagas por imagem)

| Feature | Descritor | Pipeline | Threshold | Direcção | Acc individual |
|---|---|---|---|---|---|
| F1 | p90_p10 | median_gauss | 0.2525 | gt | 90.5% |
| F2 | dark_ratio | unsharp | 0.2056 | gt | 88.5% |
| F3 | std_intensity | unsharp_gauss | 0.1283 | gt | 89.0% |
| F4 | n_cantos (t=0.01, d=2) | gauss_unsharp | 6.070 | gt | 80.7% |
| **Combinado 3/4** | votação maioritária | — | — | — | **91.7%** |

### G100 (100 vagas por imagem)

| Feature | Descritor | Pipeline | Threshold | Direcção | Acc individual |
|---|---|---|---|---|---|
| F1 | glcm_homogeneity | median_gauss | 0.5674 | **lt** | 91.1% |
| F2 | sobel_mean | gauss_median | 0.0423 | gt | 88.8% |
| F3 | prewitt_mean | median_gauss | 0.0404 | gt | 88.8% |
| F4 | n_cantos (t=0.05, d=3) | median | 0.000 | gt | 83.8% |
| **Combinado 3/4** | votação maioritária | — | — | — | **89.4%** |

**Nota sobre F1 do G100:** é a única feature com direcção `lt` — homogeneidade GLCM alta significa textura uniforme (asfalto = livre), enquanto homogeneidade baixa indica textura heterogénea (veículo = ocupado). As restantes têm direcção `gt` porque valores altos de gradiente/contraste/cantos indicam presença de veículo.

**Nota sobre F4 do G100 (threshold = 0.0):** basta detectar qualquer canto para votar OC. As vagas livres de G100 têm ROIs muito pequenas (~23×46 px) onde raramente se detecta algum canto; qualquer canto detectado é sinal fiável de veículo.

---

## Resumo dos ganhos da votação maioritária

| Parque | Melhor feature individual | Acc individual | Acc combinada 3/4 | Ganho |
|---|---|---|---|---|
| G28 | F2 glcm_contrast | 91.9% | **92.3%** | +0.4 pp |
| G40 | F1 p90_p10 | 90.5% | **91.7%** | +1.2 pp |
| G100 | F1 glcm_homogeneity | 91.1% | **89.4%** | −1.7 pp |

O G100 é o único caso onde a combinação fica abaixo da melhor feature individual. Isso deve-se ao F4 (Harris) ter apenas 83.8% de accuracy nas ROIs muito pequenas de G100 — apesar do critério de 0.1 pp ter sido satisfeito, o Harris introduz algum ruído neste parque. A diferença é pequena e o 89.4% combinado ainda é um resultado sólido.

---

## Ligação ao restante pipeline

```
passo2b_per_park.py
   → recomenda features por parque
          ↓
passo3_classificador.py
   → calibra thresholds e lógica de votação
   → gera passo3_limiares.json
          ↓
passo4_avaliacao_final.py   →  avalia no split TEST (nunca visto)
passo5_pipeline_visual.py   →  visualiza o pipeline feature a feature
passo6_app_parque.py        →  aplicação interactiva (carrega o JSON)
```

`passo3_limiares.json` é o único ficheiro que transita entre passos. Passo 4, 5 e 6 nunca vêem os dados de treino — apenas o JSON com os parâmetros calibrados.
