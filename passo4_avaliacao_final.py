# passo4_avaliacao_final.py
# Passo 4 — Avaliação FINAL no split test  (corre UMA VEZ SÓ)
# Visão por Computador — MEEC / IPB 2025-2026
# Alunos: Caio Sant'Ana Oliveira (52963) - Mauro da Silva Leme (a52965)
#
# ─────────────────────────────────────────────────────────────────────────────
# PAPEL NO PROJECTO — AVALIAÇÃO HONESTA
# ─────────────────────────────────────────────────────────────────────────────
# Este é o QUARTO passo — corre apenas UMA VEZ, no final do projecto.
#
# ⚠️  ATENÇÃO: Este script usa o split TEST — dados NUNCA vistos durante a
#     calibração. Os limiares vêm exclusivamente de passo3_limiares.json
#     (calibrados no train, verificados no valid).
#     NÃO ajustar limiares depois de ver estes resultados — isso seria
#     data leakage e invalidaria a avaliação.
#
# PORQUÊ UM SPLIT TEST SEPARADO?
#   O passo3 calibrou os limiares no split TRAIN (e verificou no VALID).
#   Se avaliássemos o sistema no mesmo conjunto onde foi calibrado, os
#   resultados seriam artificialmente optimistas (overfitting). O split
#   TEST garante uma estimativa honesta da performance em dados reais.
#
# O QUE ESTE SCRIPT AVALIA:
#   - Accuracy global (todos os parques juntos)
#   - Accuracy por parque (G28, G40, G100 separados)
#   - Precision, Recall e F1-score por parque
#   - Distribuição de accuracy por imagem (ver pior e melhor caso)
#   - Identificação das imagens com pior performance (para análise qualitativa)
#
# COMO CORRER:
#   1. Certificar que passo3_limiares.json existe (gerado pelo passo3)
#   2. python passo4_avaliacao_final.py  (demora ~3-8 min)
#   3. Ver os ficheiros PNG gerados para os gráficos de resultados
# ─────────────────────────────────────────────────────────────────────────────
#
# INPUT:
#   passo3_limiares.json       — thresholds e lógica por parque
#   dataset_parkinglot/test/   — imagens e anotações COCO
#
# OUTPUT:
#   passo4_test_predicoes.csv  — predição de cada vaga do test
#   passo4_confusion.png       — confusion matrix final
#   passo4_acuracias.png       — accuracy global + por parque
#   passo4_erros_dist.png      — distribuição de accuracy por imagem
#   passo4_erros_piores.png    — amostra das piores imagens por parque

import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, gaussian, unsharp_mask
from skimage.feature import graycomatrix, graycoprops, corner_harris, corner_peaks
from scipy.ndimage import median_filter as scipy_median

warnings.filterwarnings('ignore')

# ── Configuração ───────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot'))
JSON_PATH   = os.path.join(_SCRIPT_DIR, 'passo3_limiares.json')

CAT_LIVRE   = 1
CAT_OCUPADA = 2
N_WORST     = 4   # piores imagens a mostrar por parque
PARQUES     = ['G28', 'G40', 'G100']
CORES_PARQUE = {'G28': '#2980b9', 'G40': '#27ae60', 'G100': '#e67e22'}


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINES (idêntico ao passo3)
# ══════════════════════════════════════════════════════════════════════════════
def _unsharp(g):
    return np.clip(unsharp_mask(g, radius=2, amount=1.0), 0.0, 1.0)

PIPELINE_FUNCS = {
    'raw'          : lambda g: g,
    'gauss'        : lambda g: gaussian(g, sigma=1),
    'median'       : lambda g: scipy_median(g, size=3).astype(float),
    'unsharp'      : _unsharp,
    'median_gauss' : lambda g: gaussian(scipy_median(g, size=3), sigma=1),
    'gauss_median' : lambda g: scipy_median(gaussian(g, sigma=1), size=3).astype(float),
    'unsharp_gauss': lambda g: gaussian(_unsharp(g), sigma=1),
    'gauss_unsharp': lambda g: np.clip(
        unsharp_mask(gaussian(g, sigma=1), radius=2, amount=1.0), 0.0, 1.0),
}

# ── Configuração per-parque (idêntico ao passo3) ───────────────────────────
PARK_CFG = {
    'G28': {
        'n_vagas' : 28,
        'feats'   : [('unsharp',      'std_intensity',    '[unsharp] std_intensity'),
                     ('gauss',        'glcm_contrast',    '[gauss] glcm_contrast'),
                     ('raw',          'std_intensity',    '[raw] std_intensity')],
        'f4_pipeline': 'unsharp',  'f4_thresh': 0.05, 'f4_min_dist': 3,
    },
    'G40': {
        'n_vagas' : 40,
        'feats'   : [('median_gauss',  'p90_p10',        '[median_gauss] p90_p10'),
                     ('unsharp',       'dark_ratio',     '[unsharp] dark_ratio'),
                     ('unsharp_gauss', 'std_intensity',  '[unsharp_gauss] std_intensity')],
        'f4_pipeline': 'gauss_unsharp', 'f4_thresh': 0.01, 'f4_min_dist': 2,
    },
    'G100': {
        'n_vagas' : 100,
        'feats'   : [('median_gauss', 'glcm_homogeneity', '[median_gauss] glcm_homogeneity'),
                     ('gauss_median', 'sobel_mean',       '[gauss_median] sobel_mean'),
                     ('median_gauss', 'prewitt_mean',     '[median_gauss] prewitt_mean')],
        'f4_pipeline': 'median', 'f4_thresh': 0.05, 'f4_min_dist': 3,
    },
}
N_VAGAS_TO_PARQUE = {v['n_vagas']: p for p, v in PARK_CFG.items()}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURES (idêntico ao passo3)
# ══════════════════════════════════════════════════════════════════════════════
def _glcm_props(roi, prop):
    q64  = (np.clip(roi, 0.0, 1.0) * 63).astype(np.uint8)
    glcm = graycomatrix(q64, [1], [0], levels=64, symmetric=True, normed=True)
    return float(graycoprops(glcm, prop)[0, 0])

FEAT_FUNCS = {
    'std_intensity'   : lambda roi, **_: float(np.std(roi)),
    'p90_p10'         : lambda roi, **_: float(np.percentile(roi, 90) - np.percentile(roi, 10)),
    'dark_ratio'      : lambda roi, med_global=None, **_: float(
        np.mean(roi < max(0.02, (med_global or 0.5) - 0.10))),
    'glcm_contrast'   : lambda roi, **_: _glcm_props(roi, 'contrast'),
    'glcm_homogeneity': lambda roi, **_: _glcm_props(roi, 'homogeneity'),
    'sobel_mean'      : lambda roi, **_: float(np.mean(sobel(roi))),
    'prewitt_mean'    : lambda roi, **_: float(np.mean(prewitt(roi))),
}

def feat_n_cantos(roi, thresh, min_dist):
    try:
        pts = corner_peaks(corner_harris(roi), min_distance=min_dist, threshold_rel=thresh)
        return float(len(pts))
    except Exception:
        return 0.0


def processar_imagem(gray, parque):
    """Aplica os pipelines necessários para este parque à imagem completa.
    Igual ao passo3 — reutiliza a mesma lógica para garantir consistência."""
    cfg  = PARK_CFG[parque]
    pips = set(pp for pp, _, _ in cfg['feats']) | {cfg['f4_pipeline']}
    pp_imgs = {pp: PIPELINE_FUNCS[pp](gray) for pp in pips}
    med_globals = {}
    for pp_name, feat_name, _ in cfg['feats']:
        if feat_name == 'dark_ratio' and pp_name not in med_globals:
            med_globals[pp_name] = float(np.median(pp_imgs[pp_name]))
    return pp_imgs, med_globals


def extrair_roi(pp_imgs, med_globals, parque, ann, H, W):
    """Extrai as 4 features de uma vaga. Igual ao passo3 — mesma lógica."""
    cfg = PARK_CFG[parque]
    x, y, w, h = ann['bbox']
    x1, y1 = int(x), int(y)
    x2, y2 = min(int(x + w), W), min(int(y + h), H)
    if (y2 - y1) < 4 or (x2 - x1) < 4:
        return None
    result = {}
    for fi, (pp_name, feat_name, _) in enumerate(cfg['feats'], 1):
        roi = pp_imgs[pp_name][y1:y2, x1:x2]
        if roi.size < 16:
            return None
        result[f'f{fi}'] = FEAT_FUNCS[feat_name](roi, med_global=med_globals.get(pp_name))
    roi_f4 = pp_imgs[cfg['f4_pipeline']][y1:y2, x1:x2]
    result['f4'] = feat_n_cantos(roi_f4, cfg['f4_thresh'], cfg['f4_min_dist'])
    return result


def predizer_roi(feats_dict, park_calib):
    """
    Aplica a votação maioritária com os limiares calibrados no passo3.

    Para cada feature activa (f1, f2, f3, f4 ou só f1-f3):
      - Se direcção == 'gt': vota OC se valor > threshold
      - Se direcção == 'lt': vota OC se valor < threshold  (e.g. GLCM homogeneidade)

    Decisão final:
      votos >= min_votos  →  1 (OCUPADA)
      votos <  min_votos  →  0 (LIVRE)

    Os limiares, direcções e min_votos vêm directamente de passo3_limiares.json.
    """
    votos = 0
    for fi in park_calib['feats_usar']:
        t, d = park_calib['thresholds'][fi], park_calib['directions'][fi]
        v    = feats_dict.get(fi, 0.0)
        votos += int(v > t) if d == 'gt' else int(v < t)
    return 1 if votos >= park_calib['min_votos'] else 0


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAR COCO
# ══════════════════════════════════════════════════════════════════════════════
def carregar_coco(split):
    path = os.path.join(DATASET, split, '_annotations.coco.json')
    with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
    anns_por_img = {}
    for ann in data['annotations']:
        anns_por_img.setdefault(ann['image_id'], []).append(ann)
    img_por_id = {img['id']: img for img in data['images']}
    return data, anns_por_img, img_por_id


# ══════════════════════════════════════════════════════════════════════════════
# AVALIAÇÃO DO SPLIT TEST
# ══════════════════════════════════════════════════════════════════════════════
def avaliar_test(calib_por_parque):
    data, anns_por_img, img_por_id = carregar_coco('test')
    img_dir  = os.path.join(DATASET, 'test')
    imgs     = data['images']
    n_imgs   = len(imgs)
    linhas   = []
    t0       = time.time()

    for i, img_info in enumerate(imgs):
        if (i + 1) % 100 == 0 or i == n_imgs - 1:
            el  = time.time() - t0
            eta = el / (i + 1) * (n_imgs - i - 1)
            print(f"  test: {i+1}/{n_imgs}  ({el:.0f}s, ETA {eta:.0f}s)    ", end='\r')

        anns = anns_por_img.get(img_info['id'], [])
        if not anns:
            continue
        n_v = len(anns)
        if n_v not in N_VAGAS_TO_PARQUE:
            continue
        parque     = N_VAGAS_TO_PARQUE[n_v]
        park_calib = calib_por_parque[parque]

        gray = rgb2gray(imread(os.path.join(img_dir, img_info['file_name'])))
        H, W = gray.shape
        pp_imgs, med_globals = processar_imagem(gray, parque)

        for ann in anns:
            feats = extrair_roi(pp_imgs, med_globals, parque, ann, H, W)
            if feats is None:
                continue
            pred = predizer_roi(feats, park_calib)
            gt   = 0 if ann['category_id'] == CAT_LIVRE else 1
            x, y, w, h = ann['bbox']
            linhas.append({
                'split'     : 'test',
                'filename'  : img_info['file_name'],
                'parque'    : parque,
                'ann_id'    : ann['id'],
                'bbox_x'    : int(x), 'bbox_y': int(y),
                'bbox_w'    : int(w), 'bbox_h': int(h),
                'label_gt'  : 'livre'   if gt   == 0 else 'ocupada',
                'label_pred': 'livre'   if pred == 0 else 'ocupada',
                'correcto'  : int(pred == gt),
                'f1': round(feats.get('f1', 0.0), 6),
                'f2': round(feats.get('f2', 0.0), 6),
                'f3': round(feats.get('f3', 0.0), 6),
                'f4': round(feats.get('f4', 0.0), 6),
            })

    print()
    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
def calcular_metricas(df):
    labels = (df['label_gt'] == 'ocupada').astype(int)
    preds  = (df['label_pred'] == 'ocupada').astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    acc  = (tp + tn) / len(labels) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"\n{'═'*58}")
    print(f"  AVALIAÇÃO FINAL — TEST  ({len(df):,} vagas)")
    print(f"{'═'*58}")
    print(f"  Accuracy global : {acc:.2f}%")
    print(f"  Precision       : {prec:.2f}%")
    print(f"  Recall          : {rec:.2f}%")
    print(f"  F1-score        : {f1:.2f}%")
    print(f"\n  Confusion matrix (real ↓ / pred →):")
    print(f"              livre    ocupada")
    print(f"  livre       {tn:>6,}   {fp:>7,}")
    print(f"  ocupada     {fn:>6,}   {tp:>7,}")
    print(f"\n  Accuracy por parque:")
    for p in PARQUES:
        sub = df[df['parque'] == p]
        if sub.empty: continue
        a  = sub['correcto'].mean() * 100
        n  = len(sub)
        ok = '✅' if a >= 80 else ('⚠️' if a >= 70 else '❌')
        print(f"    {p}: {a:.2f}%  ({n:,} vagas)  {ok}")
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion(m, out_path):
    matrix = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    total  = matrix.sum()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['livre', 'ocupada'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['livre', 'ocupada'])
    ax.set_xlabel('Predição', fontsize=11)
    ax.set_ylabel('Ground Truth', fontsize=11)
    for i in range(2):
        for j in range(2):
            val  = matrix[i, j]
            perc = val / total * 100
            cor  = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f"{val:,}\n({perc:.1f}%)",
                    ha='center', va='center', color=cor, fontsize=10, fontweight='bold')
    ax.set_title(
        f"Confusion Matrix — TEST  (resultado final)\n"
        f"Accuracy: {m['acc']:.2f}%   F1: {m['f1']:.2f}%",
        fontsize=10, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


def plot_acuracias(df, m, out_path):
    parques  = PARQUES
    accs_p   = [df[df['parque']==p]['correcto'].mean()*100
                if not df[df['parque']==p].empty else 0 for p in parques]
    ns_p     = [len(df[df['parque']==p]) for p in parques]
    labels_x = ['GLOBAL'] + parques
    accs_x   = [m['acc']] + accs_p
    cores    = ['#2c3e50'] + [CORES_PARQUE[p] for p in parques]
    fig, ax  = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_x, accs_x, color=cores, edgecolor='white', linewidth=1.5, width=0.55)
    for bar, acc, n in zip(bars, accs_x, [len(df)] + ns_p):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.4,
                f"{acc:.2f}%\n({n:,})", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(80, color='green',  linestyle='--', linewidth=1.2, alpha=0.55, label='mínimo 80%')
    ax.axhline(70, color='orange', linestyle='--', linewidth=1.2, alpha=0.55, label='mínimo 70%')
    ax.set_ylim(0, 108)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(
        f"Passo 4 — Accuracy Final  (split TEST — resultado definitivo)\n"
        f"Precision: {m['prec']:.2f}%   Recall: {m['rec']:.2f}%   F1: {m['f1']:.2f}%",
        fontsize=10, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


def plot_erros_dist(df, out_path):
    """Histograma de accuracy por imagem + boxplot por parque."""
    df_imgs = (df.groupby(['parque', 'filename'])
                 .agg(acc=('correcto', 'mean'), n=('correcto', 'count'))
                 .reset_index())
    df_imgs['acc'] *= 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma por parque
    ax = axes[0]
    bins = np.linspace(0, 100, 21)
    for p in PARQUES:
        sub = df_imgs[df_imgs['parque'] == p]['acc']
        ax.hist(sub, bins=bins, alpha=0.5, color=CORES_PARQUE[p], label=p, edgecolor='white')
    ax.axvline(80, color='green',  linestyle='--', linewidth=1.5, label='80%')
    ax.axvline(70, color='orange', linestyle='--', linewidth=1.5, label='70%')
    ax.set_xlabel('Accuracy por imagem (%)', fontsize=11)
    ax.set_ylabel('Número de imagens', fontsize=11)
    ax.set_title('Distribuição de accuracy por imagem', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Boxplot por parque
    ax2 = axes[1]
    data_bp = [df_imgs[df_imgs['parque']==p]['acc'].values for p in PARQUES]
    bp = ax2.boxplot(data_bp, labels=PARQUES, patch_artist=True, notch=False)
    for patch, p in zip(bp['boxes'], PARQUES):
        patch.set_facecolor(CORES_PARQUE[p])
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    ax2.axhline(80, color='green', linestyle='--', linewidth=1.2, alpha=0.7)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Boxplot de accuracy por imagem e parque', fontsize=11, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Passo 4 — Análise de erros: distribuição de accuracy por imagem',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


def plot_piores_imagens(df, out_path):
    """Mostra as N_WORST piores imagens por parque com overlay de predições."""
    img_dir = os.path.join(DATASET, 'test')

    # Per-imagem accuracy
    df_imgs = (df.groupby(['parque', 'filename'])
                 .agg(acc=('correcto', 'mean'), n=('correcto', 'count'))
                 .reset_index())
    df_imgs['acc'] *= 100

    fig, axes = plt.subplots(len(PARQUES), N_WORST,
                              figsize=(N_WORST * 4.5, len(PARQUES) * 4))
    if len(PARQUES) == 1:
        axes = axes[np.newaxis, :]

    for row, parque in enumerate(PARQUES):
        sub_imgs = (df_imgs[df_imgs['parque'] == parque]
                    .nsmallest(N_WORST, 'acc'))

        for col in range(N_WORST):
            ax = axes[row, col]
            ax.axis('off')
            if col >= len(sub_imgs):
                continue

            fname = sub_imgs.iloc[col]['filename']
            acc_i = sub_imgs.iloc[col]['acc']
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(img_path):
                continue

            img = imread(img_path)
            ax.imshow(img)

            # Draw bboxes
            df_ann = df[(df['filename'] == fname) & (df['parque'] == parque)]
            for _, r in df_ann.iterrows():
                cor   = '#00cc00' if r['correcto'] == 1 else '#ff2222'
                rect  = mpatches.Rectangle(
                    (r['bbox_x'], r['bbox_y']), r['bbox_w'], r['bbox_h'],
                    linewidth=1.5, edgecolor=cor, facecolor='none',
                )
                ax.add_patch(rect)
                ax.text(r['bbox_x'] + r['bbox_w']/2, r['bbox_y'] - 2,
                        'L' if r['label_pred'] == 'livre' else 'O',
                        ha='center', va='bottom', fontsize=5,
                        color=cor, fontweight='bold')

            ax.set_title(f"{parque}  acc={acc_i:.0f}%\n{fname}",
                         fontsize=6.5, color=CORES_PARQUE[parque], fontweight='bold')

            if col == 0:
                ax.set_ylabel(parque, fontsize=11, fontweight='bold',
                              color=CORES_PARQUE[parque])

    legend_elem = [
        mpatches.Patch(edgecolor='#00cc00', facecolor='none', label='Correcto'),
        mpatches.Patch(edgecolor='#ff2222', facecolor='none', label='Errado'),
    ]
    fig.legend(handles=legend_elem, loc='lower center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Passo 4 — {N_WORST} piores imagens por parque\n"
        "Verde=acerto  Vermelho=erro  L=livre  O=ocupada",
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_total = time.time()

    # ── Carregar limiares do passo3 ──────────────────────────────────────────
    print(f"\nA carregar limiares de {os.path.basename(JSON_PATH)}...")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Ficheiro não encontrado: {JSON_PATH}\n"
                                "Execute primeiro o passo3_classificador.py.")
    with open(JSON_PATH, encoding='utf-8') as fh:
        limiares = json.load(fh)

    calib_por_parque = {}
    for parque in PARQUES:
        p = limiares[parque]
        calib_por_parque[parque] = {
            'parque'    : parque,
            'logica'    : p['logica'],
            'min_votos' : p['min_votos'],
            'usa_f4'    : p['usa_f4'],
            'feats_usar': p['feats_usar'],
            'thresholds': p['thresholds'],
            'directions': p['directions'],
            'acc_calib' : p['acc_calib'],
        }
        print(f"  {parque}: {p['logica']}  "
              f"{'(com F4)' if p['usa_f4'] else '(sem F4)'}  "
              f"acc_calib={p['acc_calib']:.2f}%")

    # ── Avaliação no test ────────────────────────────────────────────────────
    print(f"\nA avaliar split test  (pode demorar ~15-20 min)...\n")
    df_test = avaliar_test(calib_por_parque)

    out_csv = os.path.join(_SCRIPT_DIR, 'passo4_test_predicoes.csv')
    df_test.to_csv(out_csv, index=False)
    print(f"  CSV guardado: {os.path.basename(out_csv)}")

    # ── Métricas ─────────────────────────────────────────────────────────────
    m = calcular_metricas(df_test)

    # ── Plots ────────────────────────────────────────────────────────────────
    print(f"\nA gerar plots...")
    plot_confusion(m, os.path.join(_SCRIPT_DIR, 'passo4_confusion.png'))
    plot_acuracias(df_test, m, os.path.join(_SCRIPT_DIR, 'passo4_acuracias.png'))
    plot_erros_dist(df_test, os.path.join(_SCRIPT_DIR, 'passo4_erros_dist.png'))
    plot_piores_imagens(df_test, os.path.join(_SCRIPT_DIR, 'passo4_erros_piores.png'))

    # ── Resumo ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'═'*58}")
    print(f"  PASSO 4 CONCLUÍDO  ({elapsed/60:.1f} min)")
    print(f"{'═'*58}")
    print(f"\n  RESULTADO FINAL (TEST — definitivo para o relatório):")
    print(f"    Accuracy global : {m['acc']:.2f}%")
    print(f"    Precision       : {m['prec']:.2f}%")
    print(f"    Recall          : {m['rec']:.2f}%")
    print(f"    F1-score        : {m['f1']:.2f}%")

    ok_global  = m['acc'] >= 80
    parques_ok = []
    for p in PARQUES:
        sub = df_test[df_test['parque'] == p]
        if not sub.empty:
            a  = sub['correcto'].mean() * 100
            ok = a >= 70
            parques_ok.append(ok)

    if all(parques_ok) and ok_global:
        print(f"\n  → PASSO 4 VALIDADO. ✅")
    else:
        print(f"\n  → Resultados abaixo do critério. ❌")

    print(f"\n  Ficheiros gerados:")
    for f in ['passo4_test_predicoes.csv', 'passo4_confusion.png',
              'passo4_acuracias.png', 'passo4_erros_dist.png',
              'passo4_erros_piores.png']:
        print(f"    {f}")
