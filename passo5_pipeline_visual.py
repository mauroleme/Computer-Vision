# passo5_pipeline_visual.py
# Passo 5 — Visualização step-by-step do pipeline de classificação
# Visão por Computador — MEEC / IPB 2025-2026
# Alunos: Caio Sant'Ana Oliveira (52963) - Mauro da Silva Leme (a52965)
#
# ─────────────────────────────────────────────────────────────────────────────
# PAPEL NO PROJECTO — EXPLICAÇÃO VISUAL DO PROCESSO
# ─────────────────────────────────────────────────────────────────────────────
# Este é o QUINTO passo — corre depois do passo4 para gerar imagens
# explicativas do pipeline, úteis para o relatório e apresentação.
#
# OBJECTIVO:
#   Mostrar visualmente, passo a passo, como o classificador processa cada
#   vaga: imagem original → pré-processamento → extracção de feature → voto
#   → decisão final. Permite perceber PORQUE uma vaga foi classificada como
#   livre ou ocupada, ao ver os mapas de gradiente e as respostas Harris.
#
# ESTRUTURA DE CADA IMAGEM GERADA:
#   Linha 1: imagem completa do parque com a bbox da vaga destacada a vermelho
#   Linha 2 (por feature):
#     ROI original  |  ROI pós-pipeline  |  Mapa de resposta  |  Voto (OC/LV)
#   Linha final: barra resumo com os 4 votos e a decisão final
#
# NOTA: usa o split TEST (já avaliado) — sem risco de data leakage.
#
# COMO CORRER:
#   1. Certificar que passo3_limiares.json existe
#   2. python passo5_pipeline_visual.py
#   3. Ver imagens geradas na pasta pipeline_exemplos/
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, gaussian, unsharp_mask
from skimage.feature import (graycomatrix, graycoprops,
                              corner_harris, corner_peaks)
from scipy.ndimage import median_filter as scipy_median

warnings.filterwarnings('ignore')

# ── Configuração ───────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET       = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot'))
JSON_PATH     = os.path.join(_SCRIPT_DIR, 'passo3_limiares.json')
OUT_DIR       = os.path.join(_SCRIPT_DIR, 'pipeline_exemplos')
os.makedirs(OUT_DIR, exist_ok=True)

# Exemplos: quantos por parque × classe
N_EX_LIVRE   = 3
N_EX_OCUP    = 3
SPLIT        = 'test'    # usar test (já avaliado) — nenhum risco de data leak
SEMENTE      = 2025
PARQUES      = ['G28', 'G40', 'G100']
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

# ── Configuração per-parque ────────────────────────────────────────────────
PARK_CFG = {
    'G28': {
        'n_vagas'    : 28,
        'feats'      : [('unsharp',      'std_intensity',    '[unsharp]\nstd_intensity'),
                        ('gauss',        'glcm_contrast',    '[gauss]\nglcm_contrast'),
                        ('raw',          'std_intensity',    '[raw]\nstd_intensity')],
        'f4_pipeline': 'unsharp',
        'f4_thresh'  : 0.05,
        'f4_min_dist': 3,
        'f4_label'   : '[unsharp]\nn_cantos (t=0.05)',
    },
    'G40': {
        'n_vagas'    : 40,
        'feats'      : [('median_gauss',  'p90_p10',        '[median_gauss]\np90_p10'),
                        ('unsharp',       'dark_ratio',     '[unsharp]\ndark_ratio'),
                        ('unsharp_gauss', 'std_intensity',  '[unsharp_gauss]\nstd_intensity')],
        'f4_pipeline': 'gauss_unsharp',
        'f4_thresh'  : 0.01,
        'f4_min_dist': 2,
        'f4_label'   : '[gauss_unsharp]\nn_cantos (t=0.01)',
    },
    'G100': {
        'n_vagas'    : 100,
        'feats'      : [('median_gauss', 'glcm_homogeneity', '[median_gauss]\nglcm_homogeneity'),
                        ('gauss_median', 'sobel_mean',       '[gauss_median]\nsobel_mean'),
                        ('median_gauss', 'prewitt_mean',     '[median_gauss]\nprewitt_mean')],
        'f4_pipeline': 'median',
        'f4_thresh'  : 0.05,
        'f4_min_dist': 3,
        'f4_label'   : '[median]\nn_cantos (t=0.05)',
    },
}
N_VAGAS_TO_PARQUE = {v['n_vagas']: p for p, v in PARK_CFG.items()}


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACÇÃO DE FEATURES (com retorno dos valores intermediários para visualizar)
# ══════════════════════════════════════════════════════════════════════════════
def _glcm_props(roi, prop):
    q64  = (np.clip(roi, 0.0, 1.0) * 63).astype(np.uint8)
    glcm = graycomatrix(q64, [1], [0], levels=64, symmetric=True, normed=True)
    return float(graycoprops(glcm, prop)[0, 0])

def calcular_feat(roi, feat_name, med_global=None):
    if feat_name == 'std_intensity':
        return float(np.std(roi))
    elif feat_name == 'p90_p10':
        return float(np.percentile(roi, 90) - np.percentile(roi, 10))
    elif feat_name == 'dark_ratio':
        limiar = max(0.02, (med_global or 0.5) - 0.10)
        return float(np.mean(roi < limiar))
    elif feat_name == 'glcm_contrast':
        return _glcm_props(roi, 'contrast')
    elif feat_name == 'glcm_homogeneity':
        return _glcm_props(roi, 'homogeneity')
    elif feat_name == 'sobel_mean':
        return float(np.mean(sobel(roi)))
    elif feat_name == 'prewitt_mean':
        return float(np.mean(prewitt(roi)))
    return 0.0

def imagem_feature(roi, feat_name):
    """Imagem visual da feature (para subplot)."""
    if feat_name in ('sobel_mean',):
        return sobel(roi)
    elif feat_name in ('prewitt_mean',):
        return np.abs(prewitt(roi))
    elif feat_name == 'dark_ratio':
        return roi   # mostrar a ROI com o limiar marcado no colorbar
    else:
        return roi   # std_intensity, p90_p10, glcm → mostrar a ROI filtrada

def feat_n_cantos_detalhado(roi, thresh, min_dist):
    """Devolve (count, pontos) para visualizar os cantos detectados."""
    try:
        h    = corner_harris(roi)
        pts  = corner_peaks(h, min_distance=min_dist, threshold_rel=thresh)
        return float(len(pts)), pts
    except Exception:
        return 0.0, np.zeros((0, 2), dtype=int)


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
# SELECÇÃO DE EXEMPLOS
# ══════════════════════════════════════════════════════════════════════════════
def seleccionar_exemplos(data, anns_por_img, img_por_id, parque, n_livre, n_ocup):
    """
    Selecciona n_livre ROIs livres + n_ocup ROIs ocupadas do split.
    Devolve lista de (img_info, ann) para cada exemplo.
    """
    rng    = np.random.default_rng(SEMENTE)
    n_vg   = PARK_CFG[parque]['n_vagas']
    imgs_p = [img for img in data['images']
              if len(anns_por_img.get(img['id'], [])) == n_vg]
    rng.shuffle(imgs_p)

    livres, ocups = [], []
    for img in imgs_p:
        for ann in anns_por_img.get(img['id'], []):
            if ann['category_id'] == 1 and len(livres) < n_livre:
                livres.append((img, ann))
            elif ann['category_id'] == 2 and len(ocups) < n_ocup:
                ocups.append((img, ann))
        if len(livres) >= n_livre and len(ocups) >= n_ocup:
            break

    return livres[:n_livre], ocups[:n_ocup]


# ══════════════════════════════════════════════════════════════════════════════
# RENDERIZAR UM EXEMPLO
# ══════════════════════════════════════════════════════════════════════════════
def renderizar_exemplo(img_info, ann, parque, calib, img_dir, out_path, idx, classe):
    """
    Gera uma figura completa para um único exemplo ROI.
    Layout:
      Linha 1 (altura dupla): imagem completa + bbox destacada
      Linha 2: ROI gray | Pipeline F1 | Pipeline F2 | Pipeline F3 | F4 cantos | Decisão
    """
    cfg       = PARK_CFG[parque]
    cor       = CORES_PARQUE[parque]
    gt_label  = 'livre' if ann['category_id'] == 1 else 'ocupada'

    # Carregar imagem
    img_rgb  = imread(os.path.join(img_dir, img_info['file_name']))
    gray     = rgb2gray(img_rgb)
    H, W     = gray.shape
    x, y, w, h = ann['bbox']
    x1, y1   = int(x), int(y)
    x2, y2   = min(int(x+w), W), min(int(y+h), H)

    # Computar todos os pipelines necessários
    pips     = set(pp for pp, _, _ in cfg['feats']) | {cfg['f4_pipeline']}
    pp_imgs  = {pp: PIPELINE_FUNCS[pp](gray) for pp in pips}
    med_globals = {}
    for pp_name, feat_name, _ in cfg['feats']:
        if feat_name == 'dark_ratio' and pp_name not in med_globals:
            med_globals[pp_name] = float(np.median(pp_imgs[pp_name]))

    # Extrair features + votos
    feats_val = {}
    votos     = {}
    for fi, (pp_name, feat_name, _) in enumerate(cfg['feats'], 1):
        roi        = pp_imgs[pp_name][y1:y2, x1:x2]
        med_gl     = med_globals.get(pp_name)
        val        = calcular_feat(roi, feat_name, med_global=med_gl)
        feats_val[f'f{fi}'] = (val, roi, pp_imgs[pp_name][y1:y2, x1:x2].copy(), feat_name)
        t, d = calib['thresholds'][f'f{fi}'], calib['directions'][f'f{fi}']
        voto = int(val > t) if d == 'gt' else int(val < t)
        votos[f'f{fi}'] = (voto, t, d)

    # F4 — Harris corners
    roi_f4         = pp_imgs[cfg['f4_pipeline']][y1:y2, x1:x2]
    n_cantos, pts  = feat_n_cantos_detalhado(roi_f4, cfg['f4_thresh'], cfg['f4_min_dist'])
    t4, d4         = calib['thresholds']['f4'], calib['directions']['f4']
    voto_f4        = int(n_cantos > t4) if d4 == 'gt' else int(n_cantos < t4)
    votos['f4']    = (voto_f4, t4, d4)
    feats_val['f4'] = (n_cantos, roi_f4, roi_f4.copy(), 'n_cantos')

    # Decisão final
    total_votos = sum(votos[fi][0] for fi in calib['feats_usar'])
    pred        = 1 if total_votos >= calib['min_votos'] else 0
    pred_label  = 'ocupada' if pred == 1 else 'livre'
    correcto    = int(pred == (0 if ann['category_id'] == 1 else 1))

    # ── Layout ───────────────────────────────────────────────────────────────
    n_cols = 6   # ROI gray | F1 | F2 | F3 | F4 | Decisão
    fig    = plt.figure(figsize=(n_cols * 3.5, 10))
    gs     = GridSpec(2, n_cols, figure=fig, height_ratios=[1.8, 1],
                      hspace=0.4, wspace=0.35)

    # ── Linha 1: imagem completa ──────────────────────────────────────────
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.imshow(img_rgb)
    rect_cor  = '#00cc44' if gt_label == 'livre' else '#ff4444'
    rect      = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                     linewidth=3, edgecolor=rect_cor, facecolor='none')
    ax_full.add_patch(rect)
    ax_full.set_title(
        f"Imagem: {img_info['file_name']}  —  Parque {parque}\n"
        f"Vaga #{ann['id']}  |  GT: {gt_label}  |  Pred: {pred_label}  "
        f"{'✅' if correcto else '❌'}",
        fontsize=11, fontweight='bold', color=cor,
    )
    ax_full.axis('off')

    # ── Linha 2 — Col 0: ROI cinzento original ────────────────────────────
    ax_roi = fig.add_subplot(gs[1, 0])
    roi_gray = gray[y1:y2, x1:x2]
    ax_roi.imshow(roi_gray, cmap='gray', vmin=0, vmax=1)
    ax_roi.set_title('ROI original\n(gray)', fontsize=9, fontweight='bold')
    ax_roi.axis('off')
    for sp in ax_roi.spines.values():
        sp.set_edgecolor(cor); sp.set_linewidth(2)

    # ── Linha 2 — Col 1-3: Features F1, F2, F3 ──────────────────────────
    fi_keys = ['f1', 'f2', 'f3']
    for col_idx, fi in enumerate(fi_keys, 1):
        ax    = fig.add_subplot(gs[1, col_idx])
        val, roi_raw, roi_filt, feat_name = feats_val[fi]
        pp_name, _, pp_label_long        = cfg['feats'][col_idx - 1]
        voto_i, t_i, d_i                 = votos[fi]

        img_show = imagem_feature(roi_filt, feat_name)
        cmap     = 'hot' if 'sobel' in feat_name or 'prewitt' in feat_name else 'gray'
        ax.imshow(img_show, cmap=cmap, vmin=0, vmax=img_show.max() or 1)

        voto_str = f"voto={'OC' if voto_i == 1 else 'LV'}"
        dir_str  = f"{'>' if d_i=='gt' else '<'}{t_i:.3f}"
        ax.set_title(
            f"{pp_label_long}\nval={val:.4f}  {dir_str}\n{voto_str}",
            fontsize=7.5, fontweight='bold',
        )
        ax.axis('off')
        borda_cor = '#27ae60' if voto_i == 1 else '#7f8c8d'
        for sp in ax.spines.values():
            sp.set_edgecolor(borda_cor); sp.set_linewidth(2.5)

    # ── Linha 2 — Col 4: F4 (Harris corners) ────────────────────────────
    ax_f4 = fig.add_subplot(gs[1, 4])
    ax_f4.imshow(roi_f4, cmap='gray', vmin=0, vmax=1)
    if len(pts) > 0:
        ax_f4.scatter(pts[:, 1], pts[:, 0], s=12, c='red',
                      marker='x', linewidths=1.2, zorder=3)
    voto_f4i, t_f4, d_f4 = votos['f4']
    f4_usada = 'f4' in calib['feats_usar']
    usada_str = '(usada)' if f4_usada else '(ignorada)'
    ax_f4.set_title(
        f"{cfg['f4_label']}\ncantos={int(n_cantos)}  >{t_f4:.1f}\n"
        f"voto={'OC' if voto_f4i else 'LV'}  {usada_str}",
        fontsize=7.5, fontweight='bold',
    )
    ax_f4.axis('off')
    borda_f4 = '#27ae60' if (f4_usada and voto_f4i) else '#7f8c8d'
    for sp in ax_f4.spines.values():
        sp.set_edgecolor(borda_f4); sp.set_linewidth(2.5)

    # ── Linha 2 — Col 5: Decisão ─────────────────────────────────────────
    ax_dec = fig.add_subplot(gs[1, 5])
    ax_dec.axis('off')

    # Tabela de votos
    feats_usar  = calib['feats_usar']
    n_min       = calib['min_votos']
    n_feats_use = len(feats_usar)
    voto_total  = sum(votos[fi][0] for fi in feats_usar)

    lines = []
    for fi in ['f1', 'f2', 'f3', 'f4']:
        if fi not in votos:
            continue
        v, t, d  = votos[fi]
        usada_fi = fi in feats_usar
        marca    = '→' if usada_fi else '  '
        cor_v    = '#27ae60' if v == 1 else '#95a5a6'
        lines.append((marca, fi.upper(), '▲OC' if v == 1 else '▼LV', cor_v, usada_fi))

    y_pos = 0.85
    ax_dec.text(0.5, 0.95, 'VOTOS', ha='center', va='top',
                fontsize=9, fontweight='bold', transform=ax_dec.transAxes)
    for marca, fi_lbl, vot_lbl, v_cor, usada_fi in lines:
        alpha = 1.0 if usada_fi else 0.35
        ax_dec.text(0.08, y_pos, f"{marca} {fi_lbl}:", ha='left', va='center',
                    fontsize=8.5, alpha=alpha, transform=ax_dec.transAxes)
        ax_dec.text(0.82, y_pos, vot_lbl, ha='right', va='center',
                    fontsize=8.5, color=v_cor, fontweight='bold', alpha=alpha,
                    transform=ax_dec.transAxes)
        y_pos -= 0.17

    y_pos -= 0.05
    ax_dec.plot([0.05, 0.95], [y_pos, y_pos],
                color='gray', linewidth=1, transform=ax_dec.transAxes)
    y_pos -= 0.12

    totstr = f"{voto_total}/{n_feats_use} ≥ {n_min}"
    cor_dec = '#27ae60' if pred == 1 else '#2980b9'
    ax_dec.text(0.5, y_pos, totstr, ha='center', va='center',
                fontsize=9, fontweight='bold', transform=ax_dec.transAxes)
    y_pos -= 0.18
    ax_dec.text(0.5, y_pos,
                '🚗 OCUPADA' if pred == 1 else '🟢 LIVRE',
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=cor_dec, transform=ax_dec.transAxes)
    y_pos -= 0.18
    ax_dec.text(0.5, y_pos,
                f"GT: {gt_label}  {'✅' if correcto else '❌'}",
                ha='center', va='center', fontsize=9,
                color='black' if correcto else 'red',
                transform=ax_dec.transAxes)

    borda_dec = '#27ae60' if correcto else '#e74c3c'
    rect_dec = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle='round,pad=0.02',
        transform=ax_dec.transAxes,
        linewidth=3, edgecolor=borda_dec, facecolor='#f8f9fa',
    )
    ax_dec.add_patch(rect_dec)
    ax_dec.set_zorder(0)

    fig.suptitle(
        f"Pipeline — {parque}  |  Exemplo {idx:02d}  |  Classe real: {gt_label.upper()}",
        fontsize=12, fontweight='bold', color=cor, y=1.01,
    )
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA DE COMPARAÇÃO (grid compacto)
# ══════════════════════════════════════════════════════════════════════════════
def renderizar_comparacao(exemplos_por_parque, calib_por_parque, img_dir_test, out_path):
    """
    Grid 3 parques × 2 classes (livre | ocupada) — apenas a tira de features.
    Útil para comparar directamente os 3 parques.
    """
    n_cols  = 6   # gray | F1 | F2 | F3 | F4 | Decisão
    n_rows  = len(PARQUES) * 2   # livre + ocupada por parque
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.8))

    row = 0
    for parque in PARQUES:
        cfg    = PARK_CFG[parque]
        calib  = calib_por_parque[parque]
        cor    = CORES_PARQUE[parque]
        img_dir = img_dir_test

        for classe, exemplos in [('livre', exemplos_por_parque[parque]['livres']),
                                  ('ocupada', exemplos_por_parque[parque]['ocups'])]:
            if not exemplos:
                row += 1
                continue
            img_info, ann = exemplos[0]
            gt_label = 'livre' if ann['category_id'] == 1 else 'ocupada'

            img_rgb  = imread(os.path.join(img_dir, img_info['file_name']))
            gray     = rgb2gray(img_rgb)
            H, W     = gray.shape
            x, y, w, h = ann['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = min(int(x+w), W), min(int(y+h), H)

            pips    = set(pp for pp, _, _ in cfg['feats']) | {cfg['f4_pipeline']}
            pp_imgs = {pp: PIPELINE_FUNCS[pp](gray) for pp in pips}
            med_globals = {}
            for pp_name, feat_name, _ in cfg['feats']:
                if feat_name == 'dark_ratio' and pp_name not in med_globals:
                    med_globals[pp_name] = float(np.median(pp_imgs[pp_name]))

            feats_val, votos = {}, {}
            for fi, (pp_name, feat_name, _) in enumerate(cfg['feats'], 1):
                roi    = pp_imgs[pp_name][y1:y2, x1:x2]
                med_gl = med_globals.get(pp_name)
                val    = calcular_feat(roi, feat_name, med_global=med_gl)
                feats_val[f'f{fi}'] = (val, roi, feat_name)
                t, d   = calib['thresholds'][f'f{fi}'], calib['directions'][f'f{fi}']
                votos[f'f{fi}'] = (int(val > t) if d == 'gt' else int(val < t), t, d)

            roi_f4      = pp_imgs[cfg['f4_pipeline']][y1:y2, x1:x2]
            n_cantos, pts = feat_n_cantos_detalhado(roi_f4, cfg['f4_thresh'], cfg['f4_min_dist'])
            t4, d4      = calib['thresholds']['f4'], calib['directions']['f4']
            voto_f4     = int(n_cantos > t4) if d4 == 'gt' else int(n_cantos < t4)
            votos['f4'] = (voto_f4, t4, d4)

            total_votos = sum(votos[fi][0] for fi in calib['feats_usar'])
            pred        = 1 if total_votos >= calib['min_votos'] else 0
            pred_label  = 'ocupada' if pred else 'livre'
            correcto    = int(pred == (0 if ann['category_id'] == 1 else 1))

            # Col 0: ROI gray
            ax = axes[row, 0]
            ax.imshow(gray[y1:y2, x1:x2], cmap='gray', vmin=0, vmax=1)
            ax.set_ylabel(f"{parque}\n{classe}", fontsize=8, fontweight='bold', color=cor)
            ax.set_title('gray', fontsize=7)
            ax.axis('off')

            # Cols 1-3: F1 F2 F3
            for col_idx, fi in enumerate(['f1', 'f2', 'f3'], 1):
                val, roi_filt, feat_name = feats_val[fi]
                pp_n = cfg['feats'][col_idx - 1][0]
                img_s = imagem_feature(roi_filt, feat_name)
                cmap2 = 'hot' if 'sobel' in feat_name or 'prewitt' in feat_name else 'gray'
                axes[row, col_idx].imshow(img_s, cmap=cmap2, vmin=0, vmax=img_s.max() or 1)
                v_i, t_i, d_i = votos[fi]
                axes[row, col_idx].set_title(
                    f"[{pp_n}] {feat_name}\n{val:.3f}  {'▲OC' if v_i else '▼LV'}",
                    fontsize=6.5)
                axes[row, col_idx].axis('off')
                bc = '#27ae60' if v_i else '#bdc3c7'
                for sp in axes[row, col_idx].spines.values():
                    sp.set_edgecolor(bc); sp.set_linewidth(2)

            # Col 4: F4 cantos
            axes[row, 4].imshow(roi_f4, cmap='gray', vmin=0, vmax=1)
            if len(pts) > 0:
                axes[row, 4].scatter(pts[:, 1], pts[:, 0], s=8, c='red',
                                     marker='x', linewidths=1, zorder=3)
            axes[row, 4].set_title(f"n_cantos={int(n_cantos)}\n"
                                   f"{'▲OC' if voto_f4 else '▼LV'}", fontsize=6.5)
            axes[row, 4].axis('off')

            # Col 5: Decisão
            axes[row, 5].axis('off')
            cor_dec = '#27ae60' if pred else '#2980b9'
            axes[row, 5].text(0.5, 0.65,
                              '🚗 OCUPADA' if pred else '🟢 LIVRE',
                              ha='center', va='center', fontsize=10, fontweight='bold',
                              color=cor_dec, transform=axes[row, 5].transAxes)
            axes[row, 5].text(0.5, 0.35,
                              f"GT: {gt_label}\n{'✅' if correcto else '❌'}",
                              ha='center', va='center', fontsize=9,
                              color='black' if correcto else 'red',
                              transform=axes[row, 5].transAxes)
            bc2 = '#27ae60' if correcto else '#e74c3c'
            rect_d = mpatches.FancyBboxPatch(
                (0.05, 0.05), 0.9, 0.9, boxstyle='round,pad=0.02',
                transform=axes[row, 5].transAxes,
                linewidth=2.5, edgecolor=bc2, facecolor='#f8f9fa')
            axes[row, 5].add_patch(rect_d)

            row += 1

    fig.suptitle(
        "Comparação do pipeline por parque e classe\n"
        "gray | F1 | F2 | F3 | F4 cantos | Decisão",
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Carregar limiares
    print(f"\nA carregar limiares de {os.path.basename(JSON_PATH)}...")
    with open(JSON_PATH, encoding='utf-8') as fh:
        limiares = json.load(fh)

    calib_por_parque = {}
    for p in PARQUES:
        d = limiares[p]
        calib_por_parque[p] = {
            'logica'    : d['logica'],
            'min_votos' : d['min_votos'],
            'usa_f4'    : d['usa_f4'],
            'feats_usar': d['feats_usar'],
            'thresholds': d['thresholds'],
            'directions': d['directions'],
        }
        print(f"  {p}: {d['logica']}  acc_calib={d['acc_calib']:.2f}%")

    # Carregar COCO do split
    print(f"\nA carregar anotações do split '{SPLIT}'...")
    data, anns_por_img, img_por_id = carregar_coco(SPLIT)
    img_dir = os.path.join(DATASET, SPLIT)

    # Gerar exemplos por parque
    print(f"\nA gerar exemplos (pasta: {OUT_DIR})...")
    exemplos_por_parque = {}

    for parque in PARQUES:
        print(f"\n  {parque}:")
        livres, ocups = seleccionar_exemplos(
            data, anns_por_img, img_por_id, parque,
            N_EX_LIVRE, N_EX_OCUP,
        )
        exemplos_por_parque[parque] = {'livres': livres, 'ocups': ocups}

        for idx, (img_info, ann) in enumerate(livres, 1):
            out_f = os.path.join(OUT_DIR, f"{parque}_ex{idx:02d}_livre.png")
            renderizar_exemplo(img_info, ann, parque, calib_por_parque[parque],
                               img_dir, out_f, idx, 'livre')

        for idx, (img_info, ann) in enumerate(ocups, 1):
            out_f = os.path.join(OUT_DIR, f"{parque}_ex{idx:02d}_ocupada.png")
            renderizar_exemplo(img_info, ann, parque, calib_por_parque[parque],
                               img_dir, out_f, idx, 'ocupada')

    # Grid de comparação
    print(f"\n  Figura de comparação entre parques...")
    out_comp = os.path.join(OUT_DIR, 'comparacao_parques.png')
    renderizar_comparacao(exemplos_por_parque, calib_por_parque, img_dir, out_comp)

    print(f"\n{'═'*55}")
    print(f"  PASSO 5 — Pipeline visual concluído!")
    print(f"  Pasta: {OUT_DIR}")
    total = len(PARQUES) * (N_EX_LIVRE + N_EX_OCUP) + 1
    print(f"  {total} ficheiros PNG gerados.")
    print(f"{'═'*55}")
