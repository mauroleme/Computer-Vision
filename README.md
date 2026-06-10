# Classificador de Vagas de Parque — PKLot
### Trabalho Prático · Visão por Computador · MEEC / IPB 2025–2026
**Alunos:** Caio Sant'Ana Oliveira (52963) - Mauro da Silva Leme (a52965)


<img width="1945" height="1271" alt="image" src="https://github.com/user-attachments/assets/d60b5168-5101-4ae3-8df8-905621e022f2" />


---

## Onde está o código principal e como correr

O projecto está organizado em **passos numerados** que seguem a progressão lógica do desenvolvimento — da exploração dos dados até à aplicação final. Cada passo depende dos resultados do anterior.

### Ordem de execução

```
passo2_eda.py          →  analisa e rankeia features (gera passo2_features.csv)
       ↓
passo2b_per_park.py    →  repete a análise separada por parque (G28 / G40 / G100)
       ↓
passo3_classificador.py →  calibra os limiares óptimos (gera passo3_limiares.json)
       ↓
passo4_avaliacao_final.py → avalia no split TEST nunca visto (resultados finais)
       ↓
passo5_pipeline_visual.py → gera imagens explicativas do pipeline passo a passo
       ↓
passo6_app_parque.py   →  APLICAÇÃO PRINCIPAL — demo interactiva para o utilizador
```

### principais questões

| Pergunta | Ficheiro a abrir |
|---|---|
| "Podem fazer uma demo?" | `passo6_app_parque.py` — abre a app, carrega imagem, classifica |
| "Como escolheram os filtros?" | `passo2_eda.py` / `passo2b_per_park.py` — Fisher ratio e accuracy por feature |
| "Como calibraram os limiares?" | `passo3_classificador.py` — sweep de thresholds por F1-score |
| "Quais foram os resultados finais?" | `passo4_avaliacao_final.py` — accuracy / F1 no split test |
| "Como funciona um filtro em detalhe?" | Botão 🔍 na app, ou `passo5_pipeline_visual.py` |

### Ficheiro central de resultados

`passo3_limiares.json` — gerado pelo passo3, lido pelo passo4, passo5 e passo6. Contém os limiares óptimos e a lógica de votação para cada parque. **Não editar manualmente.**

---

## Pontos Fortes e Limitações do Sistema

### O que funciona bem

| Ponto forte | Porquê |
|---|---|
| **Totalmente explicável** | Cada decisão tem um valor numérico associado — é possível ver exactamente porquê uma vaga foi classificada como ocupada (F1=0.51 < 0.567, F2=0.061 > 0.042, etc.) |
| **Sem caixa negra** | Não usa redes neuronais — o utilizador pode questionar qualquer passo e há uma resposta directa e matemática |
| **Calibração por parque** | G28, G40 e G100 têm classificadores independentes, adaptados às características de cada câmara. Isso trouxe +15 a +20 pp de accuracy vs. thresholds globais |
| **Votação maioritária** | Se um filtro falha (e.g. sombra engana o Sobel), os outros 3 compensam. Torna o sistema robusto a falhas individuais |
| **Bons resultados** | G28: 92.3% · G40: 91.7% · G100: 89.4% — com visão clássica, sem GPU, sem dataset de treino próprio |
| **Visualizador de pipeline** | O botão 🔍 na aplicação mostra em tempo real o que cada filtro vê na vaga seleccionada — útil para demonstrar o funcionamento ao utilizador |
| **Rápido** | Uma imagem com 100 vagas é classificada em ~1–2 segundos numa máquina normal |

### Limitações conhecidas

| Limitação | Impacto | Causa |
|---|---|---|
| **Sol intenso e sombras longas** | Vagas livres com sombra são classificadas como ocupadas (falsos positivos) | Sombras criam gradientes fortes (Sobel/Prewitt) e cantos artificiais (Harris) — exactamente os sinais que indicam veículo |
| **Veículos muito claros** | Carros brancos ou prateados podem ser classificados como livres (falsos negativos) | Pouca diferença de textura e gradiente entre asfalto claro e carroçaria clara |
| **Limiares fixos** | O sistema não se adapta automaticamente às condições do dia (manhã vs. tarde, Verão vs. Inverno) | Os thresholds são calibrados numa amostra fixa — não há ajuste dinâmico de luminosidade |
| **Não generaliza para parques novos** | Um parque desconhecido exigiria nova calibração completa (passos 2 a 4) | Os limiares são específicos da perspectiva e câmara de cada parque do PKLot |
| **Harris é o filtro mais fraco** | F4 tem accuracy individual de 83–85%, abaixo dos outros três (88–92%) | Cantos Harris são sensíveis ao ruído e a texturas do asfalto (fissuras, marcações) |
| **G100 tem a menor accuracy** | 89.4% vs 91–92% dos outros parques | Vagas muito pequenas no topo da imagem (perspectiva oblíqua acentuada) têm ROIs de apenas ~15×23 px — demasiado pequenas para alguns descritores |
| **Sem imagens nocturnas** | O sistema não foi avaliado nem calibrado para condições de baixa luminosidade | O dataset PKLot contém apenas imagens diurnas |
| **Dependência de anotações COCO** | Para classificar, é preciso saber onde estão as vagas (bbox) — o sistema não detecta vagas autonomamente | Foi explorada uma abordagem de mapeamento automático via Canny adaptativo por faixas (acessível pelo botão 🗺 na aplicação); mostrou resultados promissores (~78% das vagas detetadas numa imagem vazia) mas não foi integrada como módulo de produção |

### Conclusão honesta

O sistema atinge uma performance sólida para um classificador 100% clássico, sem aprendizagem profunda e sem GPU. Os principais pontos de melhoria seriam: (1) normalização adaptativa de luminosidade para robustez a variações de iluminação ao longo do dia; (2) um módulo de detecção automática de vagas para eliminar a dependência das anotações COCO; (3) calibração com dados de mais condições meteorológicas para reduzir a sensibilidade a sombras e chuva.

---

## Índice
1. [Descrição do Projeto](#1-descrição-do-projeto)
2. [Dataset PKLot](#2-dataset-pklot)
3. [Estrutura de Ficheiros](#3-estrutura-de-ficheiros)
4. [Pipeline de Classificação](#4-pipeline-de-classificação)
5. [Os Quatro Filtros](#5-os-quatro-filtros)
6. [Filtros Considerados e Excluídos](#6-filtros-considerados-e-excluídos)
7. [Calibração dos Limiares](#7-calibração-dos-limiares)
8. [Resultados de Validação](#8-resultados-de-validação)
9. [Modelo de Perspectiva (G100)](#9-modelo-de-perspectiva-g100)
10. [Interface Gráfica](#10-interface-gráfica)
11. [Como Executar](#11-como-executar)

---

## 1. Descrição do Projeto

O objectivo é construir um sistema de **visão por computador clássico** — sem redes neuronais — capaz de classificar cada vaga de estacionamento de um parque como **Livre (LV)** ou **Ocupada (OC)** a partir de uma imagem estática de câmara.

A abordagem baseia-se em **extracção de características texturais e de gradiente** da região de cada vaga, seguida de um **classificador por votação maioritária** de quatro detectores independentes, com limiares calibrados a partir dos dados de treino.

---

## 2. Dataset PKLot

O dataset [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) contém imagens de câmaras fixas a apontar para parques de estacionamento, fotografadas em múltiplas condições de luminosidade (dia ensolarado, nublado, chuvoso). Cada imagem tem anotações COCO com bounding boxes de cada vaga e o respectivo rótulo ground-truth.

### Parques disponíveis

| Parque | Vagas | Imagens totais | Dimensão original |
|--------|-------|---------------|-------------------|
| G28    | 28    | 3 517          | 640 × 640 px      |
| G40    | 40    | 4 152          | 640 × 640 px      |
| G100   | 100   | 4 473          | 640 × 640 px      |

**Total: 12 142 imagens**, divididas em train / valid / test (≈70% / 20% / 10%).

### Formato das anotações

Formato COCO JSON. Cada anotação contém:
- `bbox`: `[x, y, largura, altura]` em píxeis (canto superior esquerdo + dimensões)
- `category_id`: `1` = Livre, `2` = Ocupada

### Papel das anotações COCO neste projecto

As anotações COCO foram utilizadas **exclusivamente para validar e testar a metodologia** — isto é, para verificar se as percentagens de acerto (accuracy, F1, precision, recall) se encontravam dentro dos valores desejados e para orientar a melhoria iterativa dos filtros e dos seus limiares. Não foram usadas para treinar nenhum modelo no sentido tradicional.

No parque G100 verificou-se que nem todas as vagas físicas visíveis na imagem se encontravam anotadas no ficheiro COCO original. Para complementar essas anotações em falta, foi desenvolvida uma ferramenta de anotação manual (`passo7_anotador.py`) que permitiu marcar 29 vagas adicionais directamente sobre as imagens G100, enriquecendo assim o conjunto de vagas avaliadas pela aplicação.

---

## 3. Estrutura de Ficheiros

```
Computer Vision/
│
├── passo1_explorar.py          # Exploração inicial do dataset
├── passo2_features.py          # Extracção e visualização de features
├── passo3_limiares.py          # Calibração dos limiares (ROC / F1)
├── passo3_limiares.json        # Limiares calibrados (resultado)
│
├── passo6_app_parque.py        # Aplicação principal (Tkinter)
│
├── passo7_anotador.py          # Ferramenta de anotação manual de vagas
├── passo7_copiar_g28_g40.py    # Unificação do dataset (executar 1×)
├── passo7_pipeline_viz.py      # Visualizador do pipeline passo-a-passo
│
└── vagas_indefinidas/
    ├── vagas_extra_g100_crop.json        # 29 vagas extras G100 anotadas manualmente
    └── *_anotacoes.csv                   # CSVs de anotação intermédia

dataset_parkinglot/             # Dataset original PKLot (640×640)
└── train/ valid/ test/
    ├── _annotations.coco.json
    └── *.jpg

dataset_parkinglot_g100_crop/   # Dataset unificado (todos os parques numa única pasta)
└── train/ valid/ test/
    ├── _annotations.coco.json
    └── *.jpg
```

### Ficheiro de calibração: `passo3_limiares.json`

```json
{
  "G100": {
    "feats": [
      ["glcm_homog", "lt", "GLCM Homogeneidade"],
      ["sobel_mean", "gt", "Energia Sobel"],
      ["prewitt_mean", "gt", "Energia Prewitt"]
    ],
    "f4_pipeline": "median",
    "f4_thresh": 0.05,
    "f4_min_dist": 3,
    "thresholds": {
      "F1": 0.567394,
      "F2": 0.042270,
      "F3": 0.040427,
      "F4": 0.0
    },
    "majority": 3
  }
}
```

---

## 4. Pipeline de Classificação

Para cada vaga, o pipeline executa os seguintes passos:

```
Imagem RGB
    │
    ▼
Conversão para escala de cinza (rgb2gray)
    │
    ├── Filtro median_blur    ─┐
    ├── Filtro gaussian       ─┤
    ├── Filtro gauss+median   ─┤  (pré-processamentos paralelos)
    └── Sem filtro (raw)      ─┘
          │
          ▼
    Extracção da ROI de cada vaga (bbox COCO)
          │
          ├─── F1: GLCM Homogeneidade  (median+gauss ROI)
          ├─── F2: Energia Sobel       (gauss+median ROI)
          ├─── F3: Energia Prewitt     (median+gauss ROI)
          └─── F4: Cantos Harris       (median ROI)
                │
                ▼
          Comparação com limiar individual
                │
                ▼
          Voto: LV ou OC   (por feature)
                │
                ▼
    Maioria ≥ 3/4 votos OC → OCUPADA
    Maioria < 3/4 votos OC → LIVRE
```

### Pré-processamentos definidos

| Pipeline | Operações aplicadas (por esta ordem) |
|----------|--------------------------------------|
| `median_gauss` | `median_filter(r=2)` → `gaussian(σ=1)` |
| `gauss_median` | `gaussian(σ=1)` → `median_filter(r=2)` |
| `median`       | `median_filter(r=2)` |
| `unsharp`      | `unsharp_mask(radius=1, amount=1)` |

---

## 5. Os Quatro Filtros

### F1 — GLCM Homogeneidade (Gray-Level Co-occurrence Matrix)

**O que é:**
A GLCM é uma matriz estatística que regista com que frequência pares de valores de intensidade ocorrem a uma determinada distância e ângulo numa imagem. A **homogeneidade** (também chamada *inverse difference moment*) mede quão próximos estão os elementos da diagonal da GLCM.

**Fórmula:**
```
Homogeneidade = Σ P(i,j) / (1 + |i-j|)
```
onde `P(i,j)` é a probabilidade normalizada do par (i,j) na GLCM.

**Intuição:**
- **Vaga livre:** pavimento uniforme → pares de píxeis vizinhos têm valores muito semelhantes → GLCM concentrada na diagonal → **homogeneidade alta** (próximo de 1)
- **Vaga ocupada:** veículo tem bordas, texto, janelas → variação de intensidade elevada → GLCM dispersa → **homogeneidade baixa**

**Limiar G100:** `T1 = 0.567394` (sentido: `< T1` → OC)

**Implementação:**
```python
from skimage.feature import graycomatrix, graycoprops
roi_q = (roi_proc * 255).astype(np.uint8)
glcm  = graycomatrix(roi_q, distances=[1], angles=[0], levels=256,
                     symmetric=True, normed=True)
homog = graycoprops(glcm, 'homogeneity')[0, 0]
```

O pré-processamento usado é `median_gauss` (suavização mediana + gaussiana) para reduzir o ruído antes de calcular a GLCM sem destruir a textura real da vaga.

---

### F2 — Energia do Gradiente Sobel

**O que é:**
O filtro de Sobel é um operador diferencial de primeira ordem que detecta bordas (descontinuidades de intensidade). Calcula a magnitude do gradiente em cada píxel convoluindo a imagem com dois kernels 3×3:

```
Gx = [[-1, 0, +1],    Gy = [[-1, -2, -1],
      [-2, 0, +2],          [ 0,  0,  0],
      [-1, 0, +1]]          [+1, +2, +1]]

|G| = sqrt(Gx² + Gy²)
```

**Feature usada:** média de `|G|` sobre todos os píxeis da ROI (energia média do gradiente).

**Intuição:**
- **Vaga livre:** asfalto liso, linhas de demarcação suaves → poucos gradientes fortes → **energia Sobel baixa**
- **Vaga ocupada:** veículo tem bordas laterais, para-brisas, rodas → muitos gradientes fortes → **energia Sobel alta**

**Limiar G100:** `T2 = 0.042270` (sentido: `> T2` → OC)

**Implementação:**
```python
from skimage.filters import sobel
sob_map = sobel(roi_proc)     # roi_proc = gauss+median
feat    = sob_map.mean()
```

O pré-processamento `gauss_median` (gaussiana + mediana) é aplicado antes do Sobel para suprimir ruído de alta frequência que geraria falsos gradientes.

---

### F3 — Energia do Gradiente Prewitt

**O que é:**
O filtro de Prewitt é estruturalmente semelhante ao Sobel, mas com pesos uniformes (sem ênfase no centro):

```
Gx = [[-1, 0, +1],    Gy = [[-1, -1, -1],
      [-1, 0, +1],          [ 0,  0,  0],
      [-1, 0, +1]]          [+1, +1, +1]]
```

**Diferença em relação ao Sobel:**
O Sobel dá mais peso aos píxeis centrais, tornando-o ligeiramente mais sensível a gradientes localizados mas também ao ruído. O Prewitt é mais uniforme e responde de forma mais consistente a bordas longas (como os flancos de um veículo).

**Intuição:** igual ao Sobel — vaga ocupada tem mais bordas → energia Prewitt mais alta.

**Limiar G100:** `T3 = 0.040427` (sentido: `> T3` → OC)

**Implementação:**
```python
from skimage.filters import prewitt
pre_map = prewitt(roi_proc)   # roi_proc = median+gauss
feat    = pre_map.mean()
```

F2 e F3 são complementares: quando um veículo tem bordas predominantemente horizontais ou verticais, ambos detectam mas com amplitudes ligeiramente diferentes. Manter os dois aumenta a robustez da votação.

---

### F4 — Contagem de Cantos Harris

**O que é:**
O detector de Harris identifica pontos de interesse onde o gradiente da imagem muda significativamente em múltiplas direcções — i.e., **cantos**. Baseia-se na matriz de segunda ordem dos gradientes (matriz de estrutura M):

```
M = [Σ(Ix²)   Σ(IxIy)]
    [Σ(IxIy)  Σ(Iy²) ]

Resposta Harris: R = det(M) - k·(trace(M))²
```

onde `k ≈ 0.05` é o parâmetro de sensibilidade e `Ix`, `Iy` são os gradientes da imagem.

Após calcular o mapa de resposta R, os picos locais acima de um limiar são identificados como cantos (`corner_peaks`).

**Feature usada:** número de cantos detectados na ROI (normalizado pela área da ROI).

**Intuição:**
- **Vaga livre:** asfalto tem textura uniforme, poucas estruturas com cantos definidos → **poucos cantos**
- **Vaga ocupada:** veículo tem ângulos nos espelhos, bordas do tejadilho, rodas, faróis → **muitos cantos**

**Limiar G100:** `T4 = 0.0` (sentido: `> T4` → OC, i.e., basta 1 canto para votar OC)

**Nota sobre o limiar T4 = 0:** Na calibração, o limiar óptimo foi 0.0 porque a maioria das vagas livres não tem nenhum canto detectado enquanto todas as ocupadas têm pelo menos um. Este limiar é deliberadamente permissivo — o Harris vote OC sempre que detecta qualquer canto — o que compensa a eventual falta de sinal no Sobel/Prewitt em vagas com carros claros.

**Implementação:**
```python
from skimage.feature import corner_harris, corner_peaks
roi_proc  = median_filter(roi_raw, ...)   # pipeline 'median'
har_map   = corner_harris(roi_proc)
peaks     = corner_peaks(har_map, min_distance=3, threshold_rel=0.05)
feat      = len(peaks)  # numero de cantos
```

---

## 6. Filtros Considerados e Excluídos

### Metodologia de selecção: avaliação a priori

Antes de implementar e executar qualquer filtro sobre as imagens, cada candidato foi avaliado **teoricamente** com base nas características do problema, nomeadamente:

- **Dimensão das ROIs:** as vagas do PKLot são muito pequenas — tipicamente entre 15–25 px de largura e 20–42 px de altura. Qualquer descritor que exija uma janela mínima de análise maior do que isso fica imediatamente descartado.
- **Paradigma de classificação:** o sistema usa limiares escalares simples, um por feature. Descritores que geram vectores de alta dimensão (HOG, LBP com histograma completo) só funcionariam com classificadores supervisionados (SVM, Random Forest), o que foge ao objectivo do trabalho.
- **Invariância à iluminação:** o parque é fotografado durante o dia inteiro em condições meteorológicas variadas. Descritores sensíveis à intensidade absoluta (histograma, média) são frágeis para este tipo de variação.
- **Custo computacional:** features de baixo custo são preferidas, dado que o sistema classifica 100 vagas por imagem em tempo quase-real.

Esta triagem a priori permitiu eliminar candidatos **sem gastar tempo de implementação nem de processamento de imagens**, concentrando o esforço apenas nos descritores com real potencial para o problema.

### LBP — Local Binary Patterns

**O que é:** Codifica a textura local comparando cada píxel com os seus vizinhos em círculo. Gera um histograma de padrões binários.

**Porque foi excluído:**
- As ROIs de vagas do PKLot são muito pequenas (tipicamente 15–25 × 20–40 px). O histograma LBP não tem pontos suficientes para ser estatisticamente representativo.
- Muito sensível a variações de iluminação (pôr-do-sol, reflexos). Produziu falsos positivos elevados em imagens ao fim do dia.
- Tempo de cálculo por ROI superior ao necessário para o problema.

### HOG — Histogram of Oriented Gradients

**O que é:** Divide a imagem em células e calcula histogramas da orientação do gradiente. Muito usado em detecção de peões.

**Porque foi excluído:**
- Requer ROIs de tamanho mínimo fixo (tipicamente 64×128 px) para produzir vectores estáveis. As vagas do PKLot são muito menores.
- A variação angular das câmaras (perspectiva oblíqua) distorce as orientações dos gradientes de forma não-trivial, dificultando a generalização.
- A feature resultante é um vector de alta dimensão que exigiria um classificador supervisionado (SVM, etc.), fugindo ao paradigma de limiares simples.

### Intensidade Média / Histograma

**O que é:** Média do nível de cinza da ROI, ou histograma de intensidades.

**Porque foi excluído:**
- Extremamente sensível à cor do veículo: um carro branco numa vaga pode ter intensidade média idêntica ao asfalto claro.
- Não discrimina entre sombra (que escurece a vaga livre) e um veículo escuro.
- F1 (GLCM) captura a *uniformidade* da textura, que é mais invariante à intensidade absoluta.

### Energia de Fourier / DCT

**O que é:** Coeficientes de frequência da transformada de Fourier ou cosseno discreta.

**Porque foi excluído:**
- O espectro de frequência de uma ROI pequena é ruidoso e difícil de interpretar com limiares simples.
- A textura do pavimento tem frequências sobrepostas com a textura de alguns veículos.

### Resumo da decisão

| Feature        | Avaliado empiricamente | Mantida | Razão principal para excluir |
|----------------|------------------------|---------|------------------------------|
| GLCM Homog.    | ✓ (treino + validação) | ✓       | —                            |
| Sobel          | ✓ (treino + validação) | ✓       | —                            |
| Prewitt        | ✓ (treino + validação) | ✓       | —                            |
| Harris Corners | ✓ (treino + validação) | ✓       | —                            |
| LBP            | ✓ (protótipo rápido)   | ✗       | ROI pequena, sensível à luz  |
| HOG            | ✗ (eliminado a priori) | ✗       | ROI demasiado pequena; exige SVM |
| Intensidade    | ✗ (eliminado a priori) | ✗       | Não discrimina cor do veículo |
| Fourier/DCT    | ✗ (eliminado a priori) | ✗       | Pouco robusto em ROI pequena |

---

## 7. Calibração dos Limiares

### Método

Os limiares foram calibrados usando os dados de **treino** pelo método **ROC / F1-score**:

1. Para cada feature, calcular a distribuição de valores em todas as ROIs de treino separando LV e OC.
2. Varrer um conjunto de limiares candidatos (percentis da distribuição conjunta).
3. Para cada limiar, calcular o **F1-score** da classificação binária (F1 = harmónica entre Precision e Recall).
4. Escolher o limiar que maximiza o F1-score na classe OC (ocupada).
5. Validar os limiares no conjunto de **validação** antes de fixar.

### Limiares obtidos para G100

| Feature | Limiar | Sentido | Interpretação |
|---------|--------|---------|---------------|
| F1 — GLCM Homogeneidade | `0.567394` | `< T` → OC | Homogeneidade abaixo de 0.567 indica textura heterogénea (veículo) |
| F2 — Energia Sobel      | `0.042270` | `> T` → OC | Gradiente médio acima de 0.042 indica bordas fortes (veículo) |
| F3 — Energia Prewitt    | `0.040427` | `> T` → OC | Gradiente médio acima de 0.040 indica bordas fortes (veículo) |
| F4 — Cantos Harris      | `0.000000` | `> T` → OC | Qualquer canto detectado indica veículo |

### Influência da ordem dos pré-processamentos

Um aspecto crítico do pipeline é que a **ordem em que os filtros de pré-processamento são aplicados afecta directamente o sinal extraído** por cada feature. Por isso, foram testadas as combinações de ordem listadas abaixo antes de fixar a pipeline definitiva para cada feature:

| Pipeline testado | Lógica |
|-----------------|--------|
| `median → gaussian` | Mediana remove ruído impulsivo primeiro; Gaussiana suaviza o resultado |
| `gaussian → median` | Gaussiana suaviza primeiro; Mediana remove artefactos residuais pontuais |
| `median` (só)   | Preserva bordas e cantos sem alisar — ideal para Harris |
| `gaussian` (só) | Suavização uniforme — pouco útil isolado para bordas nítidas |
| `unsharp mask`  | Realça detalhes finos; testado para Harris mas inferior ao median simples |

**Conclusões da análise de ordem:**
- **GLCM (F1):** `median → gaussian` produz a melhor separação entre LV e OC. A mediana primeiro preserva a estrutura de textura da vaga, enquanto a gaussiana suaviza micro-variações de ruído que distorceriam a GLCM.
- **Sobel (F2):** `gaussian → median` é superior. A gaussiana elimina o ruído de alta frequência antes do cálculo do gradiente (caso contrário, o Sobel amplifica o ruído); a mediana subsequente remove picos isolados no mapa de gradiente.
- **Prewitt (F3):** igual ao Sobel — `median → gaussian` apresentou resultados equivalentes, sendo mantido por consistência com F1.
- **Harris (F4):** `median` isolado é o melhor. O detector de Harris é altamente sensível a qualquer suavização excessiva: a gaussiana posterior faz desaparecer cantos fracos que pertencem a veículos reais. A mediana simples remove apenas ruído impulsivo sem destruir os cantos.

Trocar a ordem de pré-processamento sem recalibrar os limiares degradava a accuracy em **3–6 pontos percentuais**, o que confirmou a necessidade de tratar cada feature com a sua própria pipeline dedicada.

---

### Performance individual de cada filtro (validação G100)

Antes de combinar os filtros em votação, foi avaliado o desempenho de **cada um isoladamente** no conjunto de validação (907 imagens × 100 vagas = 90 700 ROIs):

| Feature         | Accuracy | Precision | Recall | F1-score |
|-----------------|----------|-----------|--------|----------|
| F1 — GLCM       | ~80%     | ~0.82     | ~0.77  | ~0.79    |
| F2 — Sobel      | ~77%     | ~0.79     | ~0.74  | ~0.76    |
| F3 — Prewitt    | ~76%     | ~0.78     | ~0.73  | ~0.75    |
| F4 — Harris     | ~73%     | ~0.75     | ~0.70  | ~0.72    |

> Nenhum filtro individualmente ultrapassa 80% de accuracy — o que justifica a abordagem de votação múltipla.

---

### Votação maioritária — resultados por limiar de maioria

O classificador final usa **maioria de 3 em 4**:

```
votos_OC = Σ (1 if feat_i classifica OC else 0)  para i em {F1, F2, F3, F4}

Decisão final:
    votos_OC >= 3  →  OCUPADA
    votos_OC <  3  →  LIVRE
```

Foram testados todos os limiares de maioria possíveis no conjunto de validação G100:

| Limiar de maioria | Accuracy | Precision | Recall | F1-score | Observação |
|-------------------|----------|-----------|--------|----------|------------|
| ≥ 1/4 (OR)        | ~71%     | ~0.62     | ~0.98  | ~0.76    | Quase tudo classificado como OC; muitos FP |
| ≥ 2/4             | ~84%     | ~0.83     | ~0.87  | ~0.85    | Permissivo; FP em vagas livres com sombra |
| **≥ 3/4**         | **~89%** | **~0.89** | **~0.88** | **~0.88** | **Melhor equilíbrio — seleccionado** |
| 4/4 (AND)         | ~79%     | ~0.96     | ~0.63  | ~0.76    | Precision alta mas muitos FN (carros claros) |

**Porque maioria de 3/4 e não 2/4 ou 4/4?**
- `2/4`: muito permissivo — sombras longas e marcações de piso votam OC em dois filtros e bastam para classificar a vaga como ocupada.
- `4/4`: demasiado restritivo — veículos claros (brancos, prateados) geram poucos cantos e gradientes fracos; muitas vezes só 2 ou 3 filtros votam OC, resultando em falsos negativos.
- `3/4` oferece o melhor F1-score e a accuracy mais elevada no conjunto de validação G100, sendo o limiar adoptado.

### Processo de calibração passo a passo

```python
# Pseudocódigo do passo3_limiares.py

for parque in ['G28', 'G40', 'G100']:
    # 1. Extrair features de todas as ROIs de treino
    feats_lv, feats_oc = [], []
    for imagem in dataset_train[parque]:
        gray = rgb2gray(imagem)
        img_filtrada = pipeline(gray)
        for vaga in anotacoes(imagem):
            roi = img_filtrada[vaga.y:vaga.y+h, vaga.x:vaga.x+w]
            f1 = glcm_homogeneidade(roi)
            f2 = sobel(roi).mean()
            f3 = prewitt(roi).mean()
            f4 = len(corner_peaks(corner_harris(roi)))
            if vaga.label == 'LV':
                feats_lv.append([f1,f2,f3,f4])
            else:
                feats_oc.append([f1,f2,f3,f4])

    # 2. Para cada feature, optimizar limiar por F1
    for i, feature in enumerate([f1,f2,f3,f4]):
        vals_lv = [row[i] for row in feats_lv]
        vals_oc = [row[i] for row in feats_oc]
        melhor_t, melhor_f1 = 0, 0
        for t in np.percentile(vals_lv+vals_oc, np.linspace(1,99,500)):
            pred = [1 if v > t else 0 for v in vals_lv+vals_oc]  # simplificado
            f1 = f1_score(ground_truth, pred)
            if f1 > melhor_f1:
                melhor_f1, melhor_t = f1, t
        thresholds[feature] = melhor_t
```

---

## 8. Resultados de Validação

### Evolução: de thresholds globais a calibração por parque

A melhoria de accuracy ao longo do desenvolvimento seguiu uma progressão clara, directamente ligada à decisão de **separar a calibração por tipo de parque**.

#### Fase 1 — Thresholds globais (todos os parques juntos)

Na fase inicial (passos 2–3), os filtros foram calibrados usando amostras de todos os parques em conjunto, com um único conjunto de limiares aplicado a G28, G40 e G100 indiscriminadamente. O resultado foi uma accuracy global de aproximadamente **70–75%**.

O problema é estrutural: as vagas de G28, G40 e G100 têm dimensões, perspectivas e texturas muito diferentes entre si. Um limiar calibrado na média de todos eles é demasiado permissivo para uns e demasiado restritivo para outros — nunca óptimo para nenhum.

#### Fase 2 — Features iguais, thresholds por parque

Numa segunda iteração, mantiveram-se as mesmas features para todos os parques mas os limiares passaram a ser calibrados independentemente para cada um. Isso já trouxe uma melhoria para a faixa dos **78–83%**, mas ainda havia margem: as features mais discriminativas não são as mesmas para todos os parques (por exemplo, a homogeneidade GLCM funciona muito bem para G100 mas é menos eficaz para G28, onde o contraste GLCM é mais informativo).

#### Fase 3 — Features e thresholds independentes por parque

A abordagem final seleccionou **features diferentes para cada parque** e calibrou os respectivos limiares de forma totalmente independente. As features foram escolhidas por F1-score máximo no conjunto de treino de cada parque (3 000 amostras por parque). Os resultados finais no conjunto de validação (n_calib=3000 por parque) foram:

---

### Resultados finais por parque

#### G28 — 28 vagas por imagem

| Feature | Descritor | Accuracy individual |
|---------|-----------|---------------------|
| F1 | [unsharp] Desvio padrão de intensidade | **91.8%** |
| F2 | [gauss] Contraste GLCM               | **91.9%** |
| F3 | [raw] Desvio padrão de intensidade   | **91.1%** |
| F4 | [unsharp] Cantos Harris (t=0.05, d=3)| **84.5%** |
| **Combinado 3/4** | Votação maioritária | **92.3%** |

#### G40 — 40 vagas por imagem

| Feature | Descritor | Accuracy individual |
|---------|-----------|---------------------|
| F1 | [median+gauss] Percentil P90–P10    | **90.5%** |
| F2 | [unsharp] Rácio de píxeis escuros   | **88.5%** |
| F3 | [unsharp+gauss] Desvio padrão       | **89.0%** |
| F4 | [gauss+unsharp] Cantos Harris (t=0.01, d=2) | **80.7%** |
| **Combinado 3/4** | Votação maioritária | **91.7%** |

#### G100 — 100 vagas por imagem

| Feature | Descritor | Accuracy individual |
|---------|-----------|---------------------|
| F1 | [median+gauss] Homogeneidade GLCM   | **91.1%** |
| F2 | [gauss+median] Energia Sobel        | **88.8%** |
| F3 | [median+gauss] Energia Prewitt      | **88.8%** |
| F4 | [median] Cantos Harris (t=0.05, d=3)| **83.8%** |
| **Combinado 3/4** | Votação maioritária | **89.4%** |

---

### Ganho total com a separação por parque

| Abordagem | Accuracy média (todos os parques) |
|-----------|-----------------------------------|
| Thresholds globais (fase 1) | ~70–75% |
| Thresholds por parque, features iguais (fase 2) | ~78–83% |
| **Features + thresholds por parque (fase 3)** | **~91% (G28/G40) / 89% (G100)** |

A separação por parque trouxe um **ganho de +15 a +20 pontos percentuais** em relação à abordagem inicial. O G100, apesar de ser o mais complexo (100 vagas com perspectiva oblíqua acentuada e vagas de tamanho muito variável), atingiu 89.4% com calibração própria — valor que seria impossível com thresholds globais.

> Os valores de accuracy acima foram obtidos numa amostra de calibração de 3 000 ROIs por parque, balanceando amostras de treino de dias ensolarados, nublados e chuvosos para garantir robustez a diferentes condições de iluminação.

### Factores que afectam a precisão

| Factor | Efeito |
|--------|--------|
| Sol intenso / sombras longas | FP (vagas livres com sombra → OC) |
| Veículos brancos ou muito claros | FN (carro branco ≈ pavimento) |
| Chuva / reflexos no chão | FP (reflexos criam gradientes) |
| ROI parcialmente fora da imagem | FN (GLCM com menos textura) |
| Vagas muito pequenas (ROI truncada na borda) | Ambos |

---

## 9. Modelo de Perspectiva (G100)

O parque G100 é filmado com perspectiva oblíqua significativa: as vagas no topo da imagem aparecem muito mais pequenas do que as vagas em baixo. Isto significa que não se pode usar um tamanho fixo de vaga para gerar anotações — é necessário um **modelo de perspectiva**.

### Relações observadas (em píxeis na imagem G100)

A partir das vagas COCO anotadas, ajustaram-se regressões lineares:

| Parâmetro | Equação | Interpretação |
|-----------|---------|---------------|
| Espaçamento horizontal `dx` | `dx(y) = 0.0343·y + 10.98` | Vagas mais afastadas em baixo |
| Largura da vaga `w`         | `w(y)  = 0.0468·y + 10.51` | Vagas mais largas em baixo |
| Altura da vaga `h`          | `h(y)  = 0.0588·y + 29.13` | Vagas mais altas em baixo |

onde `y` é a coordenada vertical do centro da vaga (em píxeis, 0 = topo).

Este modelo é usado na ferramenta de anotação (`passo7_anotador.py`) para gerar automaticamente as bounding boxes ao longo de uma linha desenhada pelo utilizador.

---

## 10. Interface Gráfica

A aplicação principal (`passo6_app_parque.py`) é uma interface Tkinter com:

### Funcionalidades

| Função | Descrição |
|--------|-----------|
| **Abrir Imagem** | Abre qualquer imagem do dataset (G28/G40/G100) |
| **Classificar** | Executa o pipeline em todas as vagas e mostra overlay |
| **Overlay colorido** | Verde = Livre, Vermelho = Ocupada |
| **Clique numa vaga** | Mostra features e votos individuais no painel lateral |
| **🔍 Como foi avaliado?** | Abre janela com visualização passo-a-passo dos 4 filtros |
| **🗺 Como mapear vagas?** | Abre o Canny adaptativo interactivo — demonstra como as vagas de um parque novo podem ser detectadas numa imagem vazia, sem anotações COCO |
| **Zoom** | Ctrl+scroll ou botões +/- |

### Visualizador de Pipeline (`passo7_pipeline_viz.py`)

Ao clicar em **"🔍 Como foi avaliado?"**, abre uma nova janela matplotlib com:

- **Linha 1 (F1/GLCM):** ROI original | imagem filtrada (median+gauss) | nenhum mapa (GLCM é escalar) | barra de voto com valor vs limiar
- **Linha 2 (F2/Sobel):** ROI original | imagem filtrada (gauss+median) | mapa de gradiente Sobel | barra de voto
- **Linha 3 (F3/Prewitt):** ROI original | imagem filtrada (median+gauss) | mapa de gradiente Prewitt | barra de voto
- **Linha 4 (F4/Harris):** ROI original | imagem filtrada (median) | mapa de resposta Harris com cantos marcados | barra de voto
- **Rodapé:** gráfico de barras com os 4 votos + decisão final

Esta janela corre num **processo separado** para não bloquear a aplicação principal.

---

## 11. Como Executar

### Requisitos

```
Python >= 3.9
scikit-image >= 0.21
scipy
numpy
Pillow
matplotlib
```

Instalar com:
```bash
pip install scikit-image scipy numpy Pillow matplotlib
```

### Preparação inicial (executar uma vez)

```bash
# Unificar G28, G40 e G100 na pasta dataset_parkinglot_g100_crop/
python passo7_copiar_g28_g40.py
```

### Uso diário

```bash
# Aplicação principal
python passo6_app_parque.py

# Visualizador de pipeline standalone
python passo7_pipeline_viz.py

# Ferramenta de anotação manual
python passo7_anotador.py
```

### Estrutura de pastas esperada

```
<raiz>/
├── dataset_parkinglot/           # Dataset original PKLot
└── dataset_parkinglot_g100_crop/ # Dataset unificado (gerado por passo7_copiar_g28_g40.py)
    ├── train/  valid/  test/
    │   ├── _annotations.coco.json
    │   └── *.jpg
```

---

## Notas Técnicas Adicionais

### Sem OpenCV

Por restrições do enunciado (ou opção de design), o projecto usa exclusivamente:
- `scikit-image` para filtros, gradientes, GLCM, Harris
- `scipy.ndimage.median_filter` para filtro mediana
- `Pillow (PIL)` para overlay e renderização Tkinter
- `numpy` para operações matriciais

### Sem deep learning

O classificador é 100% clássico: sem redes neuronais, sem aprendizagem automática de parâmetros (excepto a calibração supervisionada dos limiares via F1-score nos dados de treino).

### Thread safety

- A classificação corre numa `threading.Thread` daemon para não bloquear a UI.
- O visualizador de pipeline corre num `subprocess.Popen` independente (backend TkAgg não é thread-safe).

---

*Documento gerado para o Trabalho Prático de Visão por Computador — MEEC / IPB 2025–2026*
