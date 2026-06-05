# Passo 5 — Visualização do Pipeline de Classificação
**Visão por Computador · MEEC / IPB 2025–2026**
Caio Sant'Ana Oliveira (52963) · Mauro da Silva Leme (a52965)

---

## Visão geral

O Passo 5 é um script de **explicabilidade visual**. Não classifica nem avalia — gera imagens que mostram, passo a passo, como o classificador chegou à sua decisão para uma vaga específica. É útil para o relatório e para demonstrar ao professor que o sistema não é uma caixa negra.

---

## Ficheiros do Passo 5

| Ficheiro | Tipo | O que contém |
|---|---|---|
| `passo5_pipeline_visual.py` | Script | Gerador de imagens explicativas |
| `pipeline_exemplos/` | Pasta | Imagens geradas (uma por vaga analisada) |

---

## Estrutura de cada imagem gerada

Cada imagem tem o seguinte layout:

```
┌─────────────────────────────────────────────────────────────────┐
│  Linha 0: imagem completa do parque com bbox da vaga destacada  │
├──────────────────────────────────────────────────────────────────┤
│  F1: ROI original │ ROI pós-pipeline │ (GLCM é escalar)  │ Voto │
│  F2: ROI original │ ROI pós-pipeline │ Mapa Sobel         │ Voto │
│  F3: ROI original │ ROI pós-pipeline │ Mapa Prewitt       │ Voto │
│  F4: ROI original │ ROI pós-pipeline │ Mapa Harris+cantos │ Voto │
├──────────────────────────────────────────────────────────────────┤
│  Rodapé: barra com os 4 votos individuais + decisão final        │
└──────────────────────────────────────────────────────────────────┘
```

Para cada feature mostra:
- A ROI crua (sem processamento)
- A ROI após o pipeline específico dessa feature (e.g. `median_gauss` para F1 do G100)
- O mapa de resposta do descritor (gradiente Sobel, resposta Harris, etc.)
- O voto: verde = LV, vermelho = OC, com o valor numérico e o threshold

---

## Porquê este script existe

O classificador usa 4 features × 1 threshold cada = 4 números para tomar uma decisão. O Passo 5 torna esses 4 números visíveis e interpretáveis:

- Permite perceber **porque** uma vaga foi classificada incorrectamente
- Mostra que o sistema tem uma lógica clara e auditável
- Serve de suporte visual para o relatório (figuras das secções de metodologia)
- Na apresentação, permite mostrar ao professor o que cada filtro "vê"

---

## Diferença entre passo5 e passo7_pipeline_viz

| | `passo5_pipeline_visual.py` | `passo7_pipeline_viz.py` |
|---|---|---|
| **Uso** | Batch — gera imagens estáticas para o relatório | Interactivo — integrado no botão 🔍 do passo6 |
| **Input** | Configuração fixa no código | Recebe a ROI e coordenadas via subprocess do passo6 |
| **Output** | Ficheiros PNG na pasta `pipeline_exemplos/` | Janela matplotlib em tempo real |
| **Quando corre** | Uma vez, antes do relatório | Cada vez que o utilizador clica numa vaga no passo6 |

Ambos fazem a mesma coisa conceptualmente — a diferença é o modo de uso.
