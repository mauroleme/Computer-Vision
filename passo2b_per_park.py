# passo2b_per_park.py
# Passo 2b — Análise per-parque das features extraídas no passo2_eda.py v4
# Visão por Computador — MEEC / IPB 2025-2026
# Alunos: Caio Sant'Ana Oliveira (52963) - Mauro da Silva Leme (a52965)
#
# ─────────────────────────────────────────────────────────────────────────────
# PAPEL NO PROJECTO
# ─────────────────────────────────────────────────────────────────────────────
# Este é o passo 2b — corre DEPOIS do passo2_eda.py e ANTES do passo3.
#
# PROBLEMA IDENTIFICADO NO PASSO2:
#   O passo2_eda.py avaliou todas as features misturando os 3 parques (G28,
#   G40, G100). Isso distorce o Fisher ratio: a variância inter-parque
#   (ROIs de G100 são ~23×46px; G28 são ~51×65px) domina o denominador
#   e esconde features que são excelentes num parque mas fracas noutros.
#
# SOLUÇÃO — análise per-parque:
#   Repetir o ranking de features mas calculando Fisher e accuracy
#   SEPARADAMENTE para cada parque, usando apenas as ROIs desse parque.
#   Resultado: cada parque recebe as SUA lista de melhores features,
#   que são depois usadas no passo3 para calibrar classificadores independentes.
#
# RESULTADO FINAL:
#   As features recomendadas por este script são as que entram no PARK_CFG
#   do passo3_classificador.py. Ver o cabeçalho do passo3 para os valores.
#
# COMO CORRER:
#   1. Certificar que passo2_features.csv existe (gerado pelo passo2_eda.py)
#   2. python passo2b_per_park.py
# ─────────────────────────────────────────────────────────────────────────────
#
# INPUT : passo2_features.csv  (gerado pelo passo2_eda.py v4)
#         600 ROIs × 205 features, com coluna 'parque' (G28 / G40 / G100)
#
# MOTIVAÇÃO:
#   O passo2_eda.py avaliou todas as features em pool (3 parques juntos).
#   Quando se misturam parques, features como n_cantos perdem poder discriminativo
#   porque a variância entre parques (G100: ROIs 23×46px vs G28: 51×65px)
#   explode o denominador do Fisher ratio.
#   Este script avalia cada parque isoladamente → revela features park-specific.
#
# OBJECTIVOS:
#   1. Top features por parque (Fisher e accuracy calculados só com os 200 ROIs do parque)
#   2. Confirmar se n_cantos é útil per-parque (Fisher>1.0 esperado para G28/G40)
#   3. Heatmap de comparação: features × parques
#   4. Recomendação de features para cada parque no passo3
#
# NOTA ESTATÍSTICA:
#   200 ROIs por parque (100 livre + 100 ocupada) → ruído ±4-5 pp.
#   Diferenças < 2 pp devem ser tratadas como empate estatístico.
#
# OUTPUT:
#   passo2b_resultados.csv    — resultados por parque (parque, n_feat, combinacao, acc, fisher)
#   passo2b_violins.png       — top 5 features por parque (3 linhas × 5 colunas)
#   passo2b_ranking.png       — top 15 combinações por parque (3 colunas)
#   passo2b_comparacao.png    — heatmap accuracy + Fisher: features × parques
#   passo2b_recomendacoes.png — tabela visual das features eleitas por parque

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── Configuração ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(_SCRIPT_DIR, 'passo2_features.csv')
SEMENTE     = 42

TOP_INDIV       = 12   # features individuais usadas para gerar pares
TOP_POOL_TRIPLAS = 7   # pool para gerar triplas (top individuais + top pares)
N_SWEEP         = 200  # pontos no sweep de limiares

# Pipelines conhecidos (do mais longo para o mais curto — para parse_feat_name)
PIPELINE_NAMES = [
    'gauss_unsharp', 'gauss_tophat', 'gauss_median',
    'median_gauss',  'tophat_gauss', 'unsharp_gauss',
    'gauss', 'median', 'tophat', 'unsharp', 'clahe', 'raw',
]

PARQUES = ['G28', 'G40', 'G100']
CORES_PARQUE = {'G28': '#2980b9', 'G40': '#27ae60', 'G100': '#e67e22'}


# ══════════════════════════════════════════════════════════════════════
# FUNÇÕES UTILITÁRIAS (mesmas do passo2_eda.py)
# ══════════════════════════════════════════════════════════════════════
def parse_feat_name(feat):
    """Devolve (pipeline, feature_base) por longest-match."""
    for pp in sorted(PIPELINE_NAMES, key=len, reverse=True):
        if feat.startswith(pp + '_'):
            return pp, feat[len(pp) + 1:]
    return '', feat


def fisher_ratio(s_livre, s_ocup):
    denom = float(s_livre.var() + s_ocup.var())
    if denom < 1e-12:
        return 0.0
    return float((s_livre.mean() - s_ocup.mean()) ** 2 / denom)


def melhor_limiar(vals, labels):
    """Sweep N_SWEEP pontos; devolve (threshold, direction, accuracy)."""
    ts   = np.linspace(vals.min(), vals.max(), N_SWEEP)
    best = (0.0, 'gt', 0.0)
    for t in ts:
        for direc, pred in [('gt', (vals > t).astype(int)),
                             ('lt', (vals < t).astype(int))]:
            acc = float(np.mean(pred == labels))
            if acc > best[2]:
                best = (t, direc, acc)
    return best


def predizer(df_sub, feats_info, logica):
    labels     = df_sub['label_num'].values
    n          = len(feats_info)
    votos_ocup = np.zeros(len(labels), dtype=int)
    for feat, t, direc in feats_info:
        v = df_sub[feat].values
        votos_ocup += ((v > t) if direc == 'gt' else (v < t)).astype(int)
    if logica == 'and':
        pred = (votos_ocup >= 1).astype(int)
    else:  # majority
        thresh = n // 2 + 1
        pred   = (votos_ocup >= thresh).astype(int)
    return float(np.mean(pred == labels))


# ══════════════════════════════════════════════════════════════════════
# ANÁLISE COMPLETA PARA UM PARQUE
# ══════════════════════════════════════════════════════════════════════
def analisar_parque(df_park, feat_cols, parque_nome):
    """
    Corre análise individual → pares → triplas para um parque.
    Devolve (df_resultados, info_feat_dict).
    """
    labels   = df_park['label_num'].values
    info_feat = {}
    resultados = []

    # ── Fase 1: features individuais ─────────────────────────────────
    for feat in feat_cols:
        col = df_park[feat]
        if col.std() < 1e-10 or col.isnull().all():
            continue
        vals          = col.fillna(0).values
        t, direc, acc = melhor_limiar(vals, labels)
        fish          = fisher_ratio(
            df_park.loc[df_park['label_gt'] == 'livre',  feat].fillna(0),
            df_park.loc[df_park['label_gt'] == 'ocupada', feat].fillna(0),
        )
        info_feat[feat] = (t, direc)
        resultados.append({
            'parque': parque_nome, 'n_features': 1, 'logica': 'single',
            'combinacao': feat,
            'accuracy': round(acc * 100, 2),
            'fisher'  : round(fish, 4),
        })

    df_r1     = pd.DataFrame(resultados).sort_values('accuracy', ascending=False)
    top_indiv = df_r1.head(TOP_INDIV)['combinacao'].tolist()

    # ── Fase 2: pares ─────────────────────────────────────────────────
    for f1, f2 in itertools.combinations(top_indiv, 2):
        fi = [(f1, *info_feat[f1]), (f2, *info_feat[f2])]
        for logica in ('and', 'majority'):
            acc  = predizer(df_park, fi, logica)
            fish = np.mean([
                fisher_ratio(df_park[df_park['label_gt']=='livre'][f1].fillna(0),
                             df_park[df_park['label_gt']=='ocupada'][f1].fillna(0)),
                fisher_ratio(df_park[df_park['label_gt']=='livre'][f2].fillna(0),
                             df_park[df_park['label_gt']=='ocupada'][f2].fillna(0)),
            ])
            resultados.append({
                'parque': parque_nome, 'n_features': 2, 'logica': logica,
                'combinacao': f'{f1}  +  {f2}',
                'accuracy': round(acc * 100, 2), 'fisher': round(fish, 4),
            })

    # Pool para triplas: top individuais + features das melhores pares
    df_r2    = pd.DataFrame(resultados)
    top_pares_feats = []
    for _, row in (df_r2[df_r2['n_features'] == 2]
                   .sort_values('accuracy', ascending=False)
                   .head(8).iterrows()):
        for p in row['combinacao'].split('+'):
            top_pares_feats.append(p.strip())
    pool_trip = list(dict.fromkeys(
        top_indiv[:TOP_POOL_TRIPLAS] + top_pares_feats
    ))[:TOP_POOL_TRIPLAS * 2]

    # ── Fase 3: triplas ───────────────────────────────────────────────
    for f1, f2, f3 in itertools.combinations(pool_trip, 3):
        fi = [(f, *info_feat[f]) for f in [f1, f2, f3]]
        for logica in ('and', 'majority'):
            acc  = predizer(df_park, fi, logica)
            fish = float(np.mean([
                fisher_ratio(df_park[df_park['label_gt']=='livre'][f].fillna(0),
                             df_park[df_park['label_gt']=='ocupada'][f].fillna(0))
                for f in [f1, f2, f3]
            ]))
            resultados.append({
                'parque': parque_nome, 'n_features': 3, 'logica': logica,
                'combinacao': f'{f1}  +  {f2}  +  {f3}',
                'accuracy': round(acc * 100, 2), 'fisher': round(fish, 4),
            })

    df_res = (pd.DataFrame(resultados)
                .sort_values(['accuracy', 'n_features'], ascending=[False, True])
                .reset_index(drop=True))
    return df_res, info_feat


# ══════════════════════════════════════════════════════════════════════
# CARREGAR DADOS
# ══════════════════════════════════════════════════════════════════════
print("A carregar passo2_features.csv...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Não encontrado: {CSV_PATH}\n"
        "Corra primeiro o passo2_eda.py para gerar este ficheiro."
    )

df_all   = pd.read_csv(CSV_PATH)
meta_cols = ['parque', 'image_id', 'label_gt', 'label_num']
feat_cols = [c for c in df_all.columns if c not in meta_cols]
df_all[feat_cols] = df_all[feat_cols].fillna(0)

print(f"  {len(df_all)} ROIs × {len(feat_cols)} features")
print(df_all.groupby(['parque', 'label_gt']).size().rename('n').to_string())


# ══════════════════════════════════════════════════════════════════════
# ANÁLISE POR PARQUE
# ══════════════════════════════════════════════════════════════════════
print(f"\nA analisar por parque ({TOP_INDIV} top individuais → pares → triplas)...")

resultados_todos = []
info_por_parque  = {}    # {parque: info_feat_dict}
df_res_por_parque = {}   # {parque: df_resultados}

for parque in PARQUES:
    df_p = df_all[df_all['parque'] == parque].copy()
    n    = len(df_p)
    print(f"\n  {parque}: {n} ROIs ({df_p['label_gt'].value_counts().to_dict()})")

    df_res, info_feat = analisar_parque(df_p, feat_cols, parque)
    df_res_por_parque[parque] = df_res
    info_por_parque[parque]   = info_feat
    resultados_todos.append(df_res)

    best3 = df_res[df_res['n_features'] == 3].iloc[0]
    best1 = df_res[df_res['n_features'] == 1].iloc[0]
    pp, fb = parse_feat_name(best1['combinacao'])
    print(f"    Melhor individual : [{pp}] {fb}  →  {best1['accuracy']:.1f}%  "
          f"fisher={best1['fisher']:.3f}")
    print(f"    Melhor tripla     : {best3['accuracy']:.1f}%  ({best3['logica']})")
    parts = [p.strip() for p in best3['combinacao'].split('+')]
    for p in parts:
        pp2, fb2 = parse_feat_name(p)
        print(f"      [{pp2}] {fb2}")

df_todos = pd.concat(resultados_todos, ignore_index=True)
out_csv  = os.path.join(_SCRIPT_DIR, 'passo2b_resultados.csv')
df_todos.to_csv(out_csv, index=False)
print(f"\nGuardado: {os.path.basename(out_csv)}")


# ══════════════════════════════════════════════════════════════════════
# FIGURA 1 — Violin plots: top 5 features por parque
# ══════════════════════════════════════════════════════════════════════
print("\nA gerar violin plots por parque...")

N_VIOLIN = 5
rng_plot  = np.random.default_rng(SEMENTE)

fig, axes = plt.subplots(
    len(PARQUES), N_VIOLIN,
    figsize=(N_VIOLIN * 4, len(PARQUES) * 3.5)
)

for row, parque in enumerate(PARQUES):
    df_p       = df_all[df_all['parque'] == parque]
    df_r1_park = (df_res_por_parque[parque][df_res_por_parque[parque]['n_features'] == 1]
                  .head(N_VIOLIN))
    top5       = df_r1_park['combinacao'].tolist()
    cor_parque = CORES_PARQUE[parque]

    for col, feat in enumerate(top5):
        ax      = axes[row, col]
        vals_l  = df_p[df_p['label_gt'] == 'livre' ][feat].values
        vals_o  = df_p[df_p['label_gt'] == 'ocupada'][feat].values

        if len(vals_l) > 1 and len(vals_o) > 1:
            vp = ax.violinplot([vals_l, vals_o], positions=[1, 2],
                               showmedians=True, showextrema=True)
            vp['bodies'][0].set_facecolor('#2ecc71')
            vp['bodies'][0].set_alpha(0.55)
            vp['bodies'][1].set_facecolor('#e74c3c')
            vp['bodies'][1].set_alpha(0.55)
            vp['cmedians'].set_color('black')
            vp['cmedians'].set_linewidth(2)

        jitter_l = rng_plot.uniform(-0.07, 0.07, len(vals_l))
        jitter_o = rng_plot.uniform(-0.07, 0.07, len(vals_o))
        ax.scatter(np.full(len(vals_l), 1) + jitter_l, vals_l,
                   s=8, alpha=0.45, color='#27ae60')
        ax.scatter(np.full(len(vals_o), 2) + jitter_o, vals_o,
                   s=8, alpha=0.45, color='#c0392b')

        # Limiar óptimo
        if feat in info_por_parque[parque]:
            t_opt, _ = info_por_parque[parque][feat]
            ax.axhline(t_opt, color='navy', linewidth=1.5, linestyle='--', alpha=0.8)

        fish = df_r1_park[df_r1_park['combinacao'] == feat]['fisher'].values[0]
        acc  = df_r1_park[df_r1_park['combinacao'] == feat]['accuracy'].values[0]
        pp, fb = parse_feat_name(feat)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['livre', 'ocup.'], fontsize=8)
        ax.set_title(
            f"[{pp}]\n{fb}\nacc={acc:.1f}%  F={fish:.3f}",
            fontsize=7.5, fontweight='bold', color=cor_parque,
        )
        ax.grid(axis='y', alpha=0.3)

        if col == 0:
            ax.set_ylabel(parque, fontsize=11, fontweight='bold', color=cor_parque)

        # Borda colorida por parque
        for spine in ax.spines.values():
            spine.set_edgecolor(cor_parque)
            spine.set_linewidth(1.8)

fig.suptitle(
    "Passo 2b — Top 5 features individuais por parque\n"
    "Verde=livre  Vermelho=ocupada  Azul tracejado=limiar óptimo  |  200 ROIs por parque",
    fontsize=12, fontweight='bold', y=1.01,
)
plt.tight_layout()
out_v = os.path.join(_SCRIPT_DIR, 'passo2b_violins.png')
plt.savefig(out_v, dpi=150, bbox_inches='tight')
print(f"  Guardado: {os.path.basename(out_v)}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURA 2 — Ranking top 15 combinações por parque
# ══════════════════════════════════════════════════════════════════════
print("A gerar ranking por parque...")

cores_n = {1: '#3498db', 2: '#e67e22', 3: '#9b59b6'}

fig, axes = plt.subplots(1, 3, figsize=(22, 10), sharey=False)

for col, parque in enumerate(PARQUES):
    ax       = axes[col]
    df_r     = df_res_por_parque[parque].head(15).copy()
    cor_parque = CORES_PARQUE[parque]

    def fmt_label(row):
        partes = [p.strip() for p in row['combinacao'].split('+')]
        def fmt(s):
            pp, fb = parse_feat_name(s)
            return f"[{pp}] {fb}" if pp else s
        txt = '\n+ '.join(fmt(p) for p in partes)
        if row['n_features'] > 1:
            txt += f"  ({row['logica']})"
        return txt

    df_r['label'] = df_r.apply(fmt_label, axis=1)

    bars = ax.barh(
        range(len(df_r)), df_r['accuracy'],
        color=[cores_n[n] for n in df_r['n_features']],
        edgecolor='white', linewidth=0.5,
    )
    for i, (_, row) in enumerate(df_r.iterrows()):
        ax.text(row['accuracy'] + 0.3, i,
                f"{row['accuracy']:.1f}%",
                va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(range(len(df_r)))
    ax.set_yticklabels(df_r['label'], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy (%)', fontsize=10)
    ax.set_xlim(0, 110)
    ax.set_title(f"{parque} — Top 15 combinações", fontsize=12,
                 fontweight='bold', color=cor_parque)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_color(cor_parque)
    ax.spines['right'].set_color(cor_parque)
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color(cor_parque)

legend_h = [
    Patch(facecolor=cores_n[1], label='1 feature'),
    Patch(facecolor=cores_n[2], label='2 features'),
    Patch(facecolor=cores_n[3], label='3 features'),
]
fig.legend(handles=legend_h, loc='lower center', ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, 0.0))
fig.suptitle(
    "Passo 2b — Top 15 combinações por parque\n"
    "Análise independente: cada coluna usa apenas os 200 ROIs desse parque",
    fontsize=13, fontweight='bold',
)
plt.tight_layout(rect=[0, 0.04, 1, 1])
out_r = os.path.join(_SCRIPT_DIR, 'passo2b_ranking.png')
plt.savefig(out_r, dpi=150, bbox_inches='tight')
print(f"  Guardado: {os.path.basename(out_r)}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# FIGURA 3 — Heatmap de comparação: features × parques
# ══════════════════════════════════════════════════════════════════════
print("A gerar heatmap de comparação...")

# Union das top 20 individuais de cada parque
union_feats = []
for parque in PARQUES:
    top20 = (df_res_por_parque[parque][df_res_por_parque[parque]['n_features'] == 1]
             .head(20)['combinacao'].tolist())
    for f in top20:
        if f not in union_feats:
            union_feats.append(f)

# Limitar a 35 features (as mais importantes)
union_feats = union_feats[:35]

# Construir matrizes accuracy e fisher
acc_matrix   = np.zeros((len(union_feats), len(PARQUES)))
fish_matrix  = np.zeros((len(union_feats), len(PARQUES)))

for ci, parque in enumerate(PARQUES):
    df_p   = df_all[df_all['parque'] == parque]
    labels = df_p['label_num'].values
    for ri, feat in enumerate(union_feats):
        if feat not in df_p.columns or df_p[feat].std() < 1e-10:
            acc_matrix[ri, ci]  = 50.0
            fish_matrix[ri, ci] = 0.0
            continue
        vals       = df_p[feat].fillna(0).values
        _, _, acc  = melhor_limiar(vals, labels)
        fish       = fisher_ratio(
            df_p[df_p['label_gt'] == 'livre' ][feat].fillna(0),
            df_p[df_p['label_gt'] == 'ocupada'][feat].fillna(0),
        )
        acc_matrix[ri, ci]  = acc * 100
        fish_matrix[ri, ci] = fish

# Ordenar por média de accuracy
order = np.argsort(-acc_matrix.mean(axis=1))
union_feats_ord = [union_feats[i] for i in order]
acc_ord  = acc_matrix[order]
fish_ord = fish_matrix[order]

# Labels legíveis
feat_labels = []
for f in union_feats_ord:
    pp, fb = parse_feat_name(f)
    feat_labels.append(f"[{pp}] {fb}" if pp else f)

# Detectar features n_cantos para highlighting
is_cantos = ['n_cantos' in f for f in union_feats_ord]

fig, axes = plt.subplots(1, 2, figsize=(16, max(10, len(union_feats_ord) * 0.38)),
                         gridspec_kw={'width_ratios': [1, 1]})

cmap_acc  = plt.cm.RdYlGn
cmap_fish = plt.cm.YlOrRd

# Heatmap accuracy
im1 = axes[0].imshow(acc_ord, aspect='auto', cmap=cmap_acc,
                      vmin=60, vmax=100)
plt.colorbar(im1, ax=axes[0], shrink=0.6, label='Accuracy (%)')
axes[0].set_xticks(range(len(PARQUES)))
axes[0].set_xticklabels(PARQUES, fontsize=11, fontweight='bold')
axes[0].set_yticks(range(len(feat_labels)))
axes[0].set_yticklabels(feat_labels, fontsize=7.5)
axes[0].set_title('Accuracy por feature por parque', fontsize=11, fontweight='bold')

for ri in range(len(union_feats_ord)):
    for ci in range(len(PARQUES)):
        val = acc_ord[ri, ci]
        cor = 'white' if val > 85 else 'black'
        axes[0].text(ci, ri, f"{val:.0f}",
                     ha='center', va='center', fontsize=7, color=cor, fontweight='bold')
    if is_cantos[ri]:
        axes[0].get_yticklabels()[ri].set_color('#8e44ad')
        axes[0].get_yticklabels()[ri].set_fontweight('bold')

# Heatmap Fisher
im2 = axes[1].imshow(fish_ord, aspect='auto', cmap=cmap_fish,
                      vmin=0, vmax=max(3.0, fish_ord.max()))
plt.colorbar(im2, ax=axes[1], shrink=0.6, label='Fisher ratio')
axes[1].set_xticks(range(len(PARQUES)))
axes[1].set_xticklabels(PARQUES, fontsize=11, fontweight='bold')
axes[1].set_yticks(range(len(feat_labels)))
axes[1].set_yticklabels(feat_labels, fontsize=7.5)
axes[1].set_title('Fisher ratio por feature por parque', fontsize=11, fontweight='bold')

for ri in range(len(union_feats_ord)):
    for ci in range(len(PARQUES)):
        val = fish_ord[ri, ci]
        cor = 'white' if val > fish_ord.max() * 0.6 else 'black'
        axes[1].text(ci, ri, f"{val:.2f}",
                     ha='center', va='center', fontsize=7, color=cor, fontweight='bold')
    if is_cantos[ri]:
        axes[1].get_yticklabels()[ri].set_color('#8e44ad')
        axes[1].get_yticklabels()[ri].set_fontweight('bold')

fig.suptitle(
    "Passo 2b — Comparação features × parques  (features n_cantos em roxo)\n"
    "Linhas ordenadas por accuracy média. Verde=alto, Vermelho=baixo.",
    fontsize=12, fontweight='bold',
)
plt.tight_layout()
out_h = os.path.join(_SCRIPT_DIR, 'passo2b_comparacao.png')
plt.savefig(out_h, dpi=150, bbox_inches='tight')
print(f"  Guardado: {os.path.basename(out_h)}")
plt.close()


# ══════════════════════════════════════════════════════════════════════
# RELATÓRIO NO TERMINAL
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RELATÓRIO PASSO 2b — ANÁLISE PER-PARQUE")
print("=" * 70)

# Top 5 individuais por parque
print("\n── Top 5 features individuais por parque ──────────────────────────")
print(f"  {'Feature':<45} {'G28':>8} {'G40':>8} {'G100':>8}  Fisher/parque")
print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")

for feat in union_feats_ord[:20]:
    pp, fb = parse_feat_name(feat)
    label  = f"[{pp}] {fb}" if pp else feat
    accs   = [f"{acc_matrix[union_feats_ord.index(feat), ci]:.1f}%" for ci in range(3)]
    fishs  = [f"{fish_matrix[union_feats_ord.index(feat), ci]:.3f}" for ci in range(3)]
    cantos_tag = " ◄ n_cantos" if 'n_cantos' in feat else ""
    print(f"  {label:<45} {accs[0]:>8} {accs[1]:>8} {accs[2]:>8}  "
          f"F={fishs[0]} / {fishs[1]} / {fishs[2]}{cantos_tag}")

# Melhor tripla por parque
print("\n── Melhor combinação 3 features por parque ─────────────────────────")
for parque in PARQUES:
    df_r    = df_res_por_parque[parque]
    best    = df_r[df_r['n_features'] == 3].iloc[0]
    parts   = [p.strip() for p in best['combinacao'].split('+')]
    pp_parts = []
    for p in parts:
        pp2, fb2 = parse_feat_name(p)
        pp_parts.append(f"[{pp2}] {fb2}" if pp2 else p)
    print(f"\n  {parque}  →  {best['accuracy']:.1f}%  ({best['logica']})")
    for pp_feat in pp_parts:
        print(f"    • {pp_feat}")

# Análise n_cantos per-parque
print("\n── Harris corners (n_cantos) por parque ───────────────────────────")
print("  (Fisher no passo2 global era <0.9 por mistura de parques)")
cantos_feats = [f for f in union_feats_ord if 'n_cantos' in f][:8]
if cantos_feats:
    for feat in cantos_feats:
        ri = union_feats_ord.index(feat)
        _, fb = parse_feat_name(feat)
        accs  = acc_matrix[ri]
        fishs = fish_matrix[ri]
        print(f"  {fb:<25} "
              f"G28: acc={accs[0]:.1f}% F={fishs[0]:.3f}  "
              f"G40: acc={accs[1]:.1f}% F={fishs[1]:.3f}  "
              f"G100: acc={accs[2]:.1f}% F={fishs[2]:.3f}")
else:
    print("  (nenhuma feature n_cantos encontrada no top das análises)")

# Recomendação para passo3
print("\n── RECOMENDAÇÃO PARA PASSO 3 ──────────────────────────────────────")
print("  Usar classificadores separados por parque (um threshold set por parque).")
print("  Para cada parque, calibrar usando apenas as imagens desse parque.")
print()
for parque in PARQUES:
    df_r  = df_res_por_parque[parque]
    best3 = df_r[df_r['n_features'] == 3].iloc[0]
    parts = [p.strip() for p in best3['combinacao'].split('+')]
    print(f"  {parque}  ({best3['accuracy']:.1f}%):")
    for p in parts:
        pp2, fb2 = parse_feat_name(p)
        print(f"    • [{pp2}] {fb2}")

print(f"\n  Nota: ruído estatístico ≈ ±4-5 pp com 200 ROIs por parque.")
print(f"  Diferenças < 2 pp não são estatisticamente significativas.")
print(f"\n  Passo 3: calibrar cada parque com N_CALIB ROIs do train desse parque.")

print(f"\n{'─'*70}")
print("Ficheiros gerados:")
for f in [out_csv, out_v, out_r, out_h]:
    print(f"  {os.path.basename(f)}")
