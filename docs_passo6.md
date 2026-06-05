# Passo 6 — Aplicação Principal (Demo Interactiva)
**Visão por Computador · MEEC / IPB 2025–2026**
Caio Sant'Ana Oliveira (52963) · Mauro da Silva Leme (a52965)

---

## Visão geral

O Passo 6 é a **aplicação principal** — o único script que o professor precisa de correr para ver o sistema a funcionar. Integra tudo: carrega os limiares calibrados, carrega as anotações COCO, classifica as vagas e mostra o resultado visualmente numa interface Tkinter.

---

## Ficheiros do Passo 6

| Ficheiro | Usado no projecto final? | Função |
|---|---|---|
| `passo6_app_parque.py` | ✅ **Sim** | Aplicação principal — o único a entregar ao professor |
| `passo6_check.py` | ❌ Dev only | Verifica sintaxe e imports do passo6 sem abrir a janela |
| `passo6_test_logic.py` | ❌ Dev only | Testa a lógica de classificação sem interface gráfica |
| `passo6_startup_test.py` | ❌ Dev only | Testa o startup da app sem abrir o mainloop Tkinter |

Os três ficheiros `_check`, `_test_logic` e `_startup_test` foram criados durante o desenvolvimento para validar que o código funcionava correctamente sem ter de abrir a interface gráfica a cada teste. Não fazem parte do pipeline final.

---

## `passo6_app_parque.py` — o que faz

### Funcionalidades da interface

| Botão / Acção | O que faz |
|---|---|
| **📂 Abrir Imagem** | Abre qualquer .jpg do dataset (G28, G40 ou G100) |
| **▶ Classificar** | Corre o pipeline em todas as vagas → mostra overlay verde/vermelho |
| **Clique numa vaga** | Mostra as 4 features, os 4 votos e a decisão no painel direito |
| **🔍 Como foi avaliado?** | Abre `passo7_pipeline_viz.py` como subprocess — visualização detalhada |
| **🗺 Como mapear vagas?** | Abre `passo_teste_canny_interativo.py` — demonstra detecção automática |
| **Zoom / Ajustar / 100%** | Controlo de zoom na imagem |

### Como o passo6 identifica o parque

O passo6 não guarda uma lista de parques por nome de ficheiro. Identifica o parque pelo **número de vagas na imagem**:

```python
N_VAGAS_TO_PARQUE = {28: 'G28', 40: 'G40', 100: 'G100'}
```

Quando carrega o COCO JSON, conta as anotações da imagem seleccionada. Se forem 28 → G28, se forem 40 → G40, se forem 100 → G100. Simples e robusto.

### Motor de classificação

O passo6 usa exactamente o mesmo algoritmo que o passo3 e passo4. A diferença é apenas o contexto:

- **passo3**: corre em batch sobre milhares de ROIs para calibrar
- **passo4**: corre em batch para avaliar
- **passo6**: corre em tempo quase-real para uma imagem seleccionada pelo utilizador

A optimização chave: em vez de aplicar os filtros de pré-processamento ROI a ROI (o que seria lento para 100 vagas), aplica cada filtro **uma vez à imagem inteira** e depois recorta a ROI da imagem já filtrada. Isso reduz o tempo de classificação de ~30s para ~1-2s numa imagem de 100 vagas.

### Thread de classificação

A classificação corre numa `threading.Thread` daemon para não bloquear a interface Tkinter. Se corresse na thread principal, a janela ficaria congelada durante o processamento.

### Subprocess para o pipeline viz

O `passo7_pipeline_viz.py` é lançado como `subprocess.Popen` (processo separado), não como thread. Motivo: a biblioteca matplotlib com backend TkAgg não é thread-safe — se fosse lançada como thread dentro do processo Tkinter existente, causaria um RuntimeError. Com subprocess, tem o seu próprio processo Python e o seu próprio event loop.

A imagem é passada entre processos via ficheiro `.npy` temporário.

---

## Como correr

```bash
python passo6_app_parque.py
```

Requisitos:
- `passo3_limiares.json` deve existir na mesma pasta
- Dataset `dataset_parkinglot_g100_crop/` deve existir com as anotações COCO
- `passo7_pipeline_viz.py` deve existir na mesma pasta (para o botão 🔍)
