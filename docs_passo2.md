# Passo 2 — Análise Exploratória de Features (EDA)
**Visão por Computador · MEEC / IPB 2025–2026**
Caio Sant'Ana Oliveira (52963) · Mauro da Silva Leme (a52965)

---

## Visão geral

O Passo 2 não é um único script — é uma sequência de três análises que evoluíram umas das outras, cada uma respondendo a uma limitação encontrada na anterior.

```
passo2_hough_corners_test.py   →  "Harris corners funciona?"
        ↓  (resposta: sim — Fisher > 1.0 nos 3 parques)
passo2_eda.py                  →  análise global de 205 features
        ↓  (problema: misturar parques distorce o Fisher ratio)
passo2b_per_park.py            →  análise separada por parque
        ↓
passo3_classificador.py        →  calibração com as features eleitas
```

---

## Script 1 — `passo2_hough_corners_test.py`

### O que faz
Script de exploração visual e estatística para testar duas hipóteses antes de comprometer tempo de processamento com a análise completa:

- **Hough Lines:** vaga livre → marcações de chão visíveis → mais linhas detetadas; vaga ocupada → carro cobre as marcações → menos linhas
- **Harris Corners:** vaga ocupada → carro tem muitos cantos (janelas, retrovisores, bordas) → mais cantos detetados

### Metodologia
- 150 ROIs por classe (livre/ocupada) por parque, amostradas aleatoriamente do split de treino
- Calcula Fisher ratio para `n_linhas` (Hough probabilístico) e `n_cantos` (Harris)
- Gera visualizações: comparação de ROIs, mapa de cantos e linhas sobre imagem completa, violin plots

### Resultado chave
Harris corners apresentou **Fisher > 1.0** nos três parques (G28, G40, G100). Este resultado justificou a inclusão de `n_cantos` como feature candidata no passo2_eda.py v4. Hough Lines mostrou-se menos robusto (sensível à iluminação e ao estado das marcações de chão).

### Saídas
| Ficheiro | Conteúdo |
|---|---|
| `hough_corners_rois.png` | Comparação visual: 3 ROIs livres vs 3 ocupadas por parque |
| `hough_corners_full.png` | Hough + Harris Corners sobre imagem completa 640×640 |
| `hough_corners_stats.png` | Violin plots de n_linhas e n_cantos por classe e parque |

---

## Script 2 — `passo2_eda.py` (v4)

### O que faz
Análise exploratória sistemática de todas as features candidatas, combinadas com todos os pipelines de pré-processamento definidos. Testa **205 combinações** numa amostra balanceada de 600 ROIs (200 por parque, 100 livres + 100 ocupadas).

### Metodologia

#### Fisher Ratio
Métrica principal para avaliar o poder discriminativo de cada feature:

```
Fisher = (μ_OC − μ_LV)² / (σ²_OC + σ²_LV)
```

- **μ_OC, μ_LV** — médias da feature para vagas Ocupadas e Livres
- **σ²_OC, σ²_LV** — variâncias
- Fisher alto → médias afastadas + variâncias baixas → boa separabilidade

Adicionalmente, para cada feature calcula-se a **accuracy com threshold óptimo** (ponto de corte que maximiza F1-score no conjunto de calibração).

#### Pipelines testados (12)

| Pipeline | Operação |
|---|---|
| `raw` | Sem pré-processamento (referência) |
| `gauss` | Gaussiano σ=1 |
| `median` | Filtro mediana 3×3 |
| `tophat` | White top-hat disk(5) |
| `unsharp` | Unsharp mask r=2, a=1 |
| `clahe` | CLAHE (controlo — confirmado fraco) |
| `gauss_tophat` | Gaussiano → top-hat |
| `gauss_median` | Gaussiano → mediana |
| `median_gauss` | Mediana → Gaussiano |
| `tophat_gauss` | Top-hat → Gaussiano |
| `unsharp_gauss` | Unsharp → Gaussiano |
| `gauss_unsharp` | Gaussiano → Unsharp |

A ordem importa: `median_gauss` e `gauss_median` produzem resultados diferentes porque a mediana preserva bordas enquanto o Gaussiano as suaviza — aplicar primeiro um ou o outro altera o sinal extraído por cada feature.

#### Features testadas (17+)

| Feature | Descrição |
|---|---|
| `glcm_homog` | GLCM Homogeneidade |
| `glcm_contrast` | GLCM Contraste |
| `glcm_energy` | GLCM Energia |
| `glcm_corr` | GLCM Correlação |
| `sobel_mean` | Energia média do gradiente Sobel |
| `prewitt_mean` | Energia média do gradiente Prewitt |
| `laplacian_mean` | Energia média do Laplaciano |
| `std_intensity` | Desvio padrão de intensidade |
| `mean_intensity` | Média de intensidade |
| `p90_p10` | Percentil 90 − Percentil 10 |
| `dark_ratio` | Fracção de píxeis escuros |
| `entropy` | Entropia de Shannon |
| `lbp_std` | Desvio padrão do LBP |
| `hog_std` | Desvio padrão do HOG |
| `hsv_s_mean` | Saturação média (canal HSV) |
| `n_cantos` | Contagem de cantos Harris (4 variantes de parâmetros) |

**Total: 12 pipelines × 17 features ≈ 205 combinações** (n_cantos tem 4 variantes × 12 pipelines = 48 colunas adicionais).

### Limitação identificada
Ao misturar os três parques numa análise única, a variância entre parques (ROIs de G100 têm ~23×46 px; G28 têm ~51×65 px) domina o denominador do Fisher ratio. Isso mascarou features excelentes para um parque específico mas medianas para os outros. Este problema levou à criação do passo2b.

### Saídas
| Ficheiro | Conteúdo |
|---|---|
| `passo2_features.csv` | 600 ROIs × 205 features + coluna `parque` + `label` |
| `passo2_top_indiv.png` | Top 15 features individuais (Fisher ratio) |
| `passo2_top_pares.png` | Top pares de features |
| `passo2_top_triplas.png` | Top triplas de features |
| `passo2_violins_top5.png` | Distribuições das 5 melhores features (livre vs ocupada) |

---

## Script 3 — `passo2b_per_park.py`

### O que faz
Lê o CSV gerado pelo passo2_eda.py e repete toda a análise de ranking **separadamente para cada parque** (G28, G40, G100). Cada parque usa apenas as suas 200 ROIs, eliminando a variância inter-parque do Fisher ratio.

### Metodologia
- Input: `passo2_features.csv` (600 ROIs, 205 features)
- Para cada parque: filtrar as 200 ROIs do parque → recalcular Fisher e accuracy → rankear features individuais, pares e triplas
- Compara resultados entre parques num heatmap: features × parques

### Resultado chave
Cada parque revelou as suas melhores features:

| Parque | Features eleitas |
|---|---|
| G28 | `std_intensity` (unsharp), `glcm_contrast` (gauss), `std_intensity` (raw) + Harris |
| G40 | `p90_p10` (median+gauss), `dark_ratio` (unsharp), `std_intensity` (unsharp+gauss) + Harris |
| G100 | `glcm_homog` (median+gauss), `sobel_mean` (gauss+median), `prewitt_mean` (median+gauss) + Harris |

Estas recomendações são o output directo que entra no `PARK_CFG` do passo3_classificador.py.

### Nota estatística
200 ROIs por parque (100 livre + 100 ocupada) introduz ruído de ±4–5 pp. Diferenças inferiores a 2 pp entre features são tratadas como empate estatístico.

### Saídas
| Ficheiro | Conteúdo |
|---|---|
| `passo2b_resultados.csv` | Resultados por parque (parque, combinação, accuracy, Fisher) |
| `passo2b_violins.png` | Top 5 features por parque (3 linhas × 5 colunas) |
| `passo2b_ranking.png` | Top 15 combinações por parque (3 colunas) |
| `passo2b_comparacao.png` | Heatmap accuracy + Fisher: features × parques |
| `passo2b_recomendacoes.png` | Tabela visual das features eleitas por parque |

---

## Resumo do fluxo completo

```
Dataset PKLot (train split)
        │
        ▼
passo2_hough_corners_test.py
  → Confirma: Harris corners tem Fisher > 1.0 nos 3 parques
  → Decisão: incluir n_cantos como feature candidata
        │
        ▼
passo2_eda.py  (v4 — já inclui n_cantos)
  → Testa 205 combinações (pipelines × features) em pool
  → Gera passo2_features.csv
  → Problema identificado: Fisher distorcido pela mistura de parques
        │
        ▼
passo2b_per_park.py
  → Lê passo2_features.csv
  → Analisa cada parque isoladamente
  → Produz recomendações de features por parque
        │
        ▼
passo3_classificador.py
  → Usa as features recomendadas pelo passo2b para cada parque
```
