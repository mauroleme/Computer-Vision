# Passo 7 — Ferramentas Auxiliares e Utilitários
**Visão por Computador · MEEC / IPB 2025–2026**
Caio Sant'Ana Oliveira (52963) · Mauro da Silva Leme (a52965)

---

## Visão geral

O "Passo 7" não é um único passo linear — é um conjunto de ferramentas auxiliares criadas ao longo do projecto para resolver problemas específicos. Cada script tem um propósito distinto e a maioria só precisa de correr uma vez (ou foi usado apenas durante o desenvolvimento).

---

## Todos os ficheiros passo7

| Ficheiro | Usado no pipeline final? | Função resumida |
|---|---|---|
| `passo7_crop_g100.py` | ✅ Uma vez | Recorta as imagens G100 de 640×640 para 575×371 px |
| `passo7_copiar_g28_g40.py` | ✅ Uma vez | Copia G28+G40 para a pasta unificada G100 crop |
| `passo7_anotador.py` | ✅ Usado | Ferramenta de anotação semi-automática de vagas G100 |
| `passo7_canonico.py` | ✅ Usado | Consolida CSVs de anotação → `vagas_extra_g100_crop.json` |
| `passo7_pipeline_viz.py` | ✅ Integrado | Visualizador de pipeline (botão 🔍 no passo6) |
| `passo7_vagas_indefinidas.py` | ❌ Dev/análise | Varreu áreas não anotadas para encontrar vagas em falta |
| `passo7_gap_analise_v2.py` | ❌ Dev/análise | Análise detalhada do gap entre vagas 15 e 16 do G100 |
| `passo7_row_analysis.py` | ❌ Dev/análise | Análise de linhas de vagas numa imagem específica |
| `passo7_auto_grid.py` | ❌ Dev/análise | Teste de paralaxe e auto-grid de vagas não anotadas |
| `passo7_anotador_check.py` | ❌ Dev only | Verificador de sintaxe do passo7_anotador.py |

---

## Scripts do pipeline final (usados na entrega)

### `passo7_crop_g100.py` — Recorte das imagens G100

**Problema:** As imagens G100 originais são 640×640 px mas as 100 vagas anotadas só ocupam uma zona central (x=11..586, y=146..517). A zona exterior tem relva, edifícios e outros elementos que introduzem ruído nas features sem trazer informação útil.

**Solução:** Cortar todas as imagens G100 para 575×371 px, centrado nas vagas. As coordenadas das bboxes no COCO JSON são actualizadas para reflectir o novo sistema de coordenadas.

**Resultado:** Pasta `dataset_parkinglot_g100_crop/` com train/valid/test recortados.

**Quando correr:** Uma vez, antes de qualquer outro script.

---

### `passo7_copiar_g28_g40.py` — Unificação do dataset

**Problema:** O passo6 tinha de gerir três pastas separadas (dataset_parkinglot para G28/G40, dataset_parkinglot_g100_crop para G100). Isso complicava o carregamento das imagens.

**Solução:** Copiar as imagens G28 e G40 para a pasta `dataset_parkinglot_g100_crop/`, actualizando os COCO JSON para incluir essas anotações com IDs não colisivos.

**Resultado:** Uma única pasta com todas as 12.142 imagens (G28 + G40 + G100). O passo6 só precisa de apontar para um sítio.

**Quando correr:** Uma vez, depois do `passo7_crop_g100.py`.

---

### `passo7_anotador.py` — Ferramenta de anotação semi-automática

**Problema:** O dataset G100 original não tinha todas as vagas físicas anotadas no COCO JSON — algumas vagas visíveis na imagem não tinham bbox. Sem bbox, o passo6 não as classificava.

**Solução:** Ferramenta Tkinter interactiva que permite ao utilizador:
1. Abrir uma imagem G100
2. Activar o modo **Linha** (tecla `L`) e arrastar horizontalmente sobre uma fila de vagas
3. O sistema usa o modelo de perspectiva do G100 para gerar bboxes automaticamente ao longo da linha
4. Clicar em cada vaga para confirmar ou inverter o label (Livre/Ocupada)
5. Exportar para CSV (`Ctrl+S`)

**Resultado:** Ficheiros CSV em `vagas_indefinidas/` com as novas anotações.

**Teclas:**

| Tecla | Acção |
|---|---|
| `L` | Modo Linha — clicar+arrastar para gerar vagas |
| `C` | Modo Confirmar — clicar numa vaga para inverter label |
| `Ctrl+Z` | Desfazer última linha |
| `Ctrl+S` | Guardar CSV |
| `+` / `-` | Zoom |
| `F` | Ajustar zoom |

---

### `passo7_canonico.py` — Consolidação das anotações extras

**Problema:** O `passo7_anotador.py` gerava um CSV por sessão de anotação. Havia múltiplos CSVs com sobreposições e duplicados.

**Solução:** Lê todos os CSVs de `vagas_indefinidas/`, agrupa por posição (y ±25px, x ±18px) para remover duplicados e gera um único `vagas_extra_g100_crop.json` com as 29 vagas extra confirmadas.

**Resultado:** `vagas_indefinidas/vagas_extra_g100_crop.json` — carregado pelo passo6 para complementar as anotações COCO originais.

---

### `passo7_pipeline_viz.py` — Visualizador interactivo do pipeline

**Problema:** O passo5 gera imagens estáticas de pipeline. Durante a demo ao professor, é mais útil clicar numa vaga específica e ver o seu pipeline em tempo real.

**Solução:** Script matplotlib interactivo que:
- Mostra uma imagem G100 com overlay de classificação
- Ao clicar numa vaga, abre uma janela com 4 linhas (uma por feature): ROI original → ROI processada → mapa de resposta → voto
- Linha de rodapé com os 4 votos e a decisão final

**Integração:** É chamado pelo `passo6_app_parque.py` como subprocess quando o utilizador clica em **🔍 Como foi avaliado?**. A imagem e as coordenadas da ROI são passadas via ficheiro `.npy` temporário.

**Pode correr standalone:**
```bash
python passo7_pipeline_viz.py
```

---

## Scripts de desenvolvimento (não entregues como parte do pipeline)

Estes scripts foram criados durante o processo de investigação. Não fazem parte do pipeline final mas estão no repositório porque documentam o raciocínio por trás de algumas decisões.

### `passo7_vagas_indefinidas.py`
Varreu automaticamente zonas não anotadas das imagens G100 (acima e abaixo das filas conhecidas) para encontrar vagas em falta. Gerou uma lista de posições candidatas que foram depois confirmadas manualmente com o `passo7_anotador.py`.

### `passo7_gap_analise_v2.py`
Análise detalhada de um gap específico entre as vagas 15 e 16 do G100 que estava a dar erros sistemáticos. Gerou mapas de features deslizantes (F1, F2+F3, votos) para perceber a causa do erro — concluiu-se que era uma zona de transição entre duas sub-filas com perspectiva diferente.

### `passo7_row_analysis.py`
Análise rápida das filas de vagas numa imagem específica do G100. Agrupa vagas por Y (threshold 20px) e imprime estatísticas por fila. Usado para perceber a estrutura de linhas do parque antes de criar o modelo de perspectiva.

### `passo7_auto_grid.py`
Dois experimentos: (1) teste de paralaxe — verificar se uma ROI de diferentes alturas da imagem tem features diferentes; (2) auto-grid — tentar detectar vagas em zonas não anotadas através de varrimento sistemático com o classificador. Alimentou a decisão de criar o `passo7_anotador.py`.

### `passo7_anotador_check.py`
Script de verificação de sintaxe e imports do `passo7_anotador.py`. Criado para validar o código sem ter de abrir a interface gráfica durante o desenvolvimento.

---

## Fluxo dos scripts passo7 na preparação do dataset

```
passo7_crop_g100.py
   → cria dataset_parkinglot_g100_crop/ com imagens G100 recortadas
          ↓
passo7_copiar_g28_g40.py
   → copia G28+G40 para a mesma pasta unificada
          ↓
passo7_vagas_indefinidas.py  (análise)
passo7_gap_analise_v2.py     (análise)
passo7_row_analysis.py       (análise)
passo7_auto_grid.py          (análise)
   → identificam vagas em falta no G100
          ↓
passo7_anotador.py
   → utilizador anota manualmente as 29 vagas em falta → CSVs
          ↓
passo7_canonico.py
   → consolida CSVs → vagas_extra_g100_crop.json
          ↓
passo6_app_parque.py
   → carrega vagas_extra_g100_crop.json como complemento ao COCO
```

O `passo7_pipeline_viz.py` é paralelo a este fluxo — não prepara dados, é chamado em runtime pelo passo6.
