# passo3_classificador.py  v2 — classificadores per-parque
# Passo 3 — Um classificador por parque, calibrado no train, verificado no valid
# Visão por Computador — MEEC / IPB 2025-2026
# Alunos: Caio Sant'Ana Oliveira (52963) - Mauro da Silva Leme (a52965)
#
# ─────────────────────────────────────────────────────────────────────────────
# PAPEL NO PROJECTO — O CORAÇÃO DA CALIBRAÇÃO
# ─────────────────────────────────────────────────────────────────────────────
# Este é o TERCEIRO passo. É aqui que o classificador é treinado/calibrado.
# "Treino" aqui não significa redes neuronais — significa encontrar o melhor
# limiar (threshold) para cada feature, por parque, maximizando o F1-score.
#
# FLUXO INTERNO:
#   FASE 1 — Extracção de features de N_CALIB ROIs do split TRAIN por parque
#            (amostradas aleatoriamente, 50% livres / 50% ocupadas)
#   FASE 2 — Sweep de limiares: para cada feature, testar N_SWEEP valores
#            entre o mínimo e máximo observados → escolher o T que dá o
#            maior F1-score na classe OC (ocupada)
#   FASE 3 — Testar maioria 2/4 vs 3/4 → guardar a lógica que dá maior acc
#   FASE 4 — Validar no split VALID completo (todas as ROIs, não apenas amostra)
#   FASE 5 — Guardar tudo em passo3_limiares.json
#
# PORQUÊ F1-SCORE E NÃO ACCURACY?
#   O dataset PKLot tem desequilíbrio de classes (mais vagas livres que
#   ocupadas em certas condições). O F1-score penaliza igualmente falsos
#   positivos e falsos negativos, sendo mais robusto que a accuracy simples.
#
# PORQUÊ MAIORIA E NÃO UM SÓ CLASSIFICADOR?
#   Cada feature capta um aspecto diferente da vaga (textura, gradiente,
#   cantos). Nenhuma isolada ultrapassa ~92%. A votação maioritária combina
#   sinais complementares e é mais robusta a condições adversas (sol, chuva).
#
# COMO CORRER:
#   python passo3_classificador.py
#   Resultado: passo3_limiares.json (lido pelo passo4, passo5 e passo6)
# ─────────────────────────────────────────────────────────────────────────────
#
# ESTRATÉGIA (Passo 2b):
#   Classificadores independentes para G28, G40 e G100.
#   Cada parque tem as suas próprias features e limiares.
#
# FEATURES ELEITAS (Passo 2b — análise per-parque, 200 ROIs/parque):
#
#   G28  (ROIs ~51×65 px, parque UFPR04):
#     F1: std_intensity     [pipeline: unsharp]          acc=94.5%  Fisher=4.39
#     F2: glcm_contrast     [pipeline: gauss]            acc=94.0%  Fisher=3.53
#     F3: std_intensity     [pipeline: raw]              acc=94.0%  Fisher=3.55
#     F4: n_cantos t=0.05 d=3 [pipeline: unsharp]       acc=83.0%  Fisher=1.17
#
#   G40  (ROIs ~50×57 px, parque UFPR05):
#     F1: p90_p10           [pipeline: median_gauss]     acc=92.5%  Fisher=2.24
#     F2: dark_ratio        [pipeline: unsharp]          acc=92.0%  Fisher=1.95
#     F3: std_intensity     [pipeline: unsharp_gauss]    acc=91.5%  Fisher=2.66
#     F4: n_cantos t=0.01 d=2 [pipeline: gauss_unsharp] acc=90.0%  Fisher=2.07
#
#   G100 (ROIs ~23×46 px, parque PUC):
#     F1: glcm_homogeneity  [pipeline: median_gauss]     acc=96.5%  Fisher=5.53
#     F2: sobel_mean        [pipeline: gauss_median]     acc=96.5%  Fisher=4.38
#     F3: prewitt_mean      [pipeline: median_gauss]     acc=96.5%  Fisher=4.46
#     F4: n_cantos t=0.05 d=3 [pipeline: median]        acc=84.5%  Fisher=1.02
#
# LÓGICA: calibrada por parque — compara maioria 2-em-3 vs 3-em-4; escolhe a melhor
# F4 (Harris corners) incluída a pedido; descartada automaticamente se não melhorar.
#
# PROCESSO:
#   1. Extrair features de amostra do train (N_CALIB ROIs/parque)
#   2. Sweeper de limiares por parque → decidir lógica de votação
#   3. Guardar limiares em passo3_limiares.json (para Passo 4)
#   4. Medir accuracy no valid COMPLETO (143.316 vagas)
#   5. Plots e CSV de resultados
#
# OUTPUT:
#   passo3_limiares.json          — limiares per-parque (lido no Passo 4)
#   passo3_valid_predicoes.csv    — predição de cada vaga do valid
#   passo3_confusion.png          — confusion matrix global
#   passo3_acuracias.png          — accuracy global + por parque
#   passo3_calib_summary.png      — resumo de calibração por parque

import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, gaussian, unsharp_mask
from skimage.feature import graycomatrix, graycoprops, corner_harris, corner_peaks
from scipy.ndimage import median_filter as scipy_median

warnings.filterwarnings('ignore')

# ── Configuração ───────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot'))

N_CALIB   = 3000   # ROIs por parque para calibração (1500 livre + 1500 ocupada)
N_SWEEP   = 300    # pontos no sweep de limiares por feature
SEMENTE   = 42
CAT_LIVRE   = 1
CAT_OCUPADA = 2


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINES
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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO PER-PARQUE
# ══════════════════════════════════════════════════════════════════════════════
# feats : lista de (pipeline_key, feature_name, label_legível)
# f4_*  : configuração de Harris corners (F4)
PARK_CFG = {
    'G28': {
        'n_vagas' : 28,
        'feats'   : [
            ('unsharp', 'std_intensity',    '[unsharp] std_intensity'),
            ('gauss',   'glcm_contrast',    '[gauss] glcm_contrast'),
            ('raw',     'std_intensity',    '[raw] std_intensity'),
        ],
        'f4_pipeline': 'unsharp',
        'f4_thresh'  : 0.05,
        'f4_min_dist': 3,
        'f4_label'   : '[unsharp] n_cantos (t=0.05, d=3)',
    },
    'G40': {
        'n_vagas' : 40,
        'feats'   : [
            ('median_gauss',  'p90_p10',        '[median_gauss] p90_p10'),
            ('unsharp',       'dark_ratio',     '[unsharp] dark_ratio'),
            ('unsharp_gauss', 'std_intensity',  '[unsharp_gauss] std_intensity'),
        ],
        'f4_pipeline': 'gauss_unsharp',
        'f4_thresh'  : 0.01,
        'f4_min_dist': 2,
        'f4_label'   : '[gauss_unsharp] n_cantos (t=0.01, d=2)',
    },
    'G100': {
        'n_vagas' : 100,
        'feats'   : [
            ('median_gauss', 'glcm_homogeneity', '[median_gauss] glcm_homogeneity'),
            ('gauss_median', 'sobel_mean',       '[gauss_median] sobel_mean'),
            ('median_gauss', 'prewitt_mean',     '[median_gauss] prewitt_mean'),
        ],
        'f4_pipeline': 'median',
        'f4_thresh'  : 0.05,
        'f4_min_dist': 3,
        'f4_label'   : '[median] n_cantos (t=0.05, d=3)',
    },
}

N_VAGAS_TO_PARQUE = {v['n_vagas']: p for p, v in PARK_CFG.items()}
PARQUES           = list(PARK_CFG.keys())
CORES_PARQUE      = {'G28': '#2980b9', 'G40': '#27ae60', 'G100': '#e67e22'}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURES — funções por nome
# ══════════════════════════════════════════════════════════════════════════════
def _glcm_props(roi, prop):
    q64  = (np.clip(roi, 0.0, 1.0) * 63).astype(np.uint8)
    glcm = graycomatrix(q64, distances=[1], angles=[0],
                        levels=64, symmetric=True, normed=True)
    return float(graycoprops(glcm, prop)[0, 0])


def feat_std_intensity(roi, med_global=None):
    return float(np.std(roi))

def feat_p90_p10(roi, med_global=None):
    return float(np.percentile(roi, 90) - np.percentile(roi, 10))

def feat_dark_ratio(roi, med_global=None):
    limiar = max(0.02, (med_global if med_global is not None else 0.5) - 0.10)
    return float(np.mean(roi < limiar))

def feat_glcm_contrast(roi, med_global=None):
    return _glcm_props(roi, 'contrast')

def feat_glcm_homogeneity(roi, med_global=None):
    return _glcm_props(roi, 'homogeneity')

def feat_sobel_mean(roi, med_global=None):
    return float(np.mean(sobel(roi)))

def feat_prewitt_mean(roi, med_global=None):
    return float(np.mean(prewitt(roi)))

FEAT_FUNCS = {
    'std_intensity'   : feat_std_intensity,
    'p90_p10'         : feat_p90_p10,
    'dark_ratio'      : feat_dark_ratio,
    'glcm_contrast'   : feat_glcm_contrast,
    'glcm_homogeneity': feat_glcm_homogeneity,
    'sobel_mean'      : feat_sobel_mean,
    'prewitt_mean'    : feat_prewitt_mean,
}

def feat_n_cantos(roi, thresh, min_dist):
    """Harris corner count — F4 para todos os parques."""
    try:
        pts = corner_peaks(corner_harris(roi),
                           min_distance=min_dist, threshold_rel=thresh)
        return float(len(pts))
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PROCESSAMENTO DE IMAGEM (pipelines) + EXTRACÇÃO DE ROI
# ══════════════════════════════════════════════════════════════════════════════
def processar_imagem(gray, parque):
    """
    Aplica os pré-processamentos necessários à imagem completa (escala de cinza).

    Optimização: em vez de aplicar TODOS os pipelines (8 possíveis), calcula
    apenas os que este parque precisa. Por exemplo, G100 só precisa de
    'median_gauss', 'gauss_median' e 'median' — os restantes são ignorados.

    Parâmetros
    ----------
    gray   : imagem em escala de cinza (float64, valores 0.0–1.0)
    parque : 'G28', 'G40' ou 'G100'

    Devolve
    -------
    pp_imgs    : dict {nome_pipeline: imagem_processada}
    med_globals: dict {nome_pipeline: mediana_da_imagem_completa}
                 Necessário para a feature dark_ratio, que compara cada
                 píxel da ROI com a mediana global da imagem (referência
                 de luminosidade adaptativa).
    """
    cfg  = PARK_CFG[parque]
    # Colectar apenas os pipelines necessários para F1, F2, F3 e F4
    pips = set(pp for pp, _, _ in cfg['feats']) | {cfg['f4_pipeline']}

    # Aplicar cada pipeline à imagem completa (mais eficiente do que por ROI)
    pp_imgs = {pp: PIPELINE_FUNCS[pp](gray) for pp in pips}

    # Mediana global — só necessária para dark_ratio (referência de luminosidade)
    med_globals = {}
    for pp_name, feat_name, _ in cfg['feats']:
        if feat_name == 'dark_ratio' and pp_name not in med_globals:
            med_globals[pp_name] = float(np.median(pp_imgs[pp_name]))

    return pp_imgs, med_globals


def extrair_roi(pp_imgs, med_globals, parque, ann, H, W):
    """
    Extrai as 4 features {f1, f2, f3, f4} de uma única vaga (ROI).

    O processo para cada feature:
      1. Recortar a ROI da imagem já pré-processada (mais rápido do que
         re-processar a vaga individual)
      2. Aplicar o descritor (GLCM, Sobel, Prewitt, Harris) à ROI

    Validações de segurança:
      - ROI com menos de 4 px em qualquer dimensão → descartada (None)
      - ROI com menos de 16 píxeis totais → descartada (GLCM instável)

    Devolve dict {'f1': val, 'f2': val, 'f3': val, 'f4': val}
    ou None se a ROI for demasiado pequena para ser classificada.
    """
    cfg  = PARK_CFG[parque]
    x, y, w, h = ann['bbox']
    x1, y1 = int(x), int(y)
    # Garantir que a ROI não sai fora dos limites da imagem
    x2, y2 = min(int(x + w), W), min(int(y + h), H)

    # Rejeitar ROIs demasiado pequenas (bbox parcialmente fora da imagem)
    if (y2 - y1) < 4 or (x2 - x1) < 4:
        return None

    result = {}
    for fi, (pp_name, feat_name, _) in enumerate(cfg['feats'], 1):
        roi = pp_imgs[pp_name][y1:y2, x1:x2]
        if roi.size < 16:   # GLCM precisa de pelo menos 16 píxeis
            return None
        med_gl = med_globals.get(pp_name)
        result[f'f{fi}'] = FEAT_FUNCS[feat_name](roi, med_global=med_gl)

    # F4 — Cantos Harris: pipeline e parâmetros específicos por parque
    # (parâmetros optimizados no passo2b — ver cabeçalho)
    roi_f4 = pp_imgs[cfg['f4_pipeline']][y1:y2, x1:x2]
    result['f4'] = feat_n_cantos(roi_f4, cfg['f4_thresh'], cfg['f4_min_dist'])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS — COCO
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
# FASE 1 — CALIBRAÇÃO (amostra do train)
# ══════════════════════════════════════════════════════════════════════════════
def extrair_calibracao():
    print(f"\nFASE 1 — Calibração  ({N_CALIB} ROIs/parque → {N_CALIB * len(PARK_CFG)} total)")
    data, anns_por_img, img_por_id = carregar_coco('train')
    img_dir = os.path.join(DATASET, 'train')
    rng     = np.random.default_rng(SEMENTE)

    # Agrupar imagens por parque (pelo número de anotações)
    imgs_por_parque = {p: [] for p in PARK_CFG}
    for img in data['images']:
        n_v = len(anns_por_img.get(img['id'], []))
        if n_v in N_VAGAS_TO_PARQUE:
            imgs_por_parque[N_VAGAS_TO_PARQUE[n_v]].append(img['id'])

    df_parts = []
    for parque in PARQUES:
        ids = list(imgs_por_parque[parque])
        rng.shuffle(ids)

        pool_l, pool_o = [], []
        for iid in ids:
            for ann in anns_por_img.get(iid, []):
                (pool_l if ann['category_id'] == CAT_LIVRE else pool_o).append((iid, ann))
            if len(pool_l) >= N_CALIB and len(pool_o) >= N_CALIB:
                break

        n_each = min(N_CALIB // 2, len(pool_l), len(pool_o))
        idx_l  = rng.choice(len(pool_l), n_each, replace=False)
        idx_o  = rng.choice(len(pool_o), n_each, replace=False)
        sel    = [pool_l[i] for i in idx_l] + [pool_o[i] for i in idx_o]

        # Agrupar por imagem para carregar cada imagem apenas uma vez
        por_img = {}
        for iid, ann in sel:
            por_img.setdefault(iid, []).append(ann)

        print(f"  {parque}: {len(sel)} ROIs de {len(por_img)} imagens", end='', flush=True)
        t0 = time.time()
        linhas = []

        for iid, anns_sel in por_img.items():
            img_info = img_por_id[iid]
            gray     = rgb2gray(imread(os.path.join(img_dir, img_info['file_name'])))
            H, W     = gray.shape
            pp_imgs, med_globals = processar_imagem(gray, parque)

            for ann in anns_sel:
                feats = extrair_roi(pp_imgs, med_globals, parque, ann, H, W)
                if feats is None:
                    continue
                feats['label']  = 0 if ann['category_id'] == CAT_LIVRE else 1
                feats['parque'] = parque
                linhas.append(feats)

        df_parts.append(pd.DataFrame(linhas))
        print(f"  ({time.time() - t0:.0f}s)")

    df = pd.concat(df_parts, ignore_index=True)
    print(f"  Total: {len(df)} ROIs  "
          f"({df['label'].sum()} ocupadas, {(df['label']==0).sum()} livres)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — SWEEP DE LIMIARES E CALIBRAÇÃO POR PARQUE
# ══════════════════════════════════════════════════════════════════════════════
def melhor_limiar(vals, labels):
    """
    Encontra o limiar óptimo para uma feature por varrimento (sweep).

    Para cada um dos N_SWEEP pontos equidistantes entre min e max da feature:
      - Testa classificar OC se val > t  (direcção 'gt' — e.g. Sobel, Harris)
      - Testa classificar OC se val < t  (direcção 'lt' — e.g. GLCM homogeneidade)
      - Calcula accuracy e guarda o melhor (t, direcção)

    A direcção é determinada automaticamente — não é assumida à partida.
    Por exemplo, homogeneidade GLCM alta → vaga livre → o limiar é 'lt' (OC se baixo).

    Devolve (threshold, direction, accuracy)
    """
    vals = np.asarray(vals, dtype=float)
    ts   = np.linspace(vals.min(), vals.max(), N_SWEEP)
    best = (float(ts[0]), 'gt', 0.0)
    for t in ts:
        for direc, pred in [('gt', (vals > t).astype(int)),
                             ('lt', (vals < t).astype(int))]:
            acc = float(np.mean(pred == labels))
            if acc > best[2]:
                best = (float(t), direc, acc)
    return best


def calibrar_parque(df_sub, parque):
    """
    Calibra os 4 limiares (F1–F4) e a lógica de votação para um parque.

    Processo:
      1. Para cada feature, encontrar o melhor threshold com melhor_limiar()
      2. Testar lógica 2-em-3 (F1+F2+F3, sem Harris):
           → pred OC se pelo menos 2 dos 3 filtros votam OC
      3. Testar lógica 3-em-4 (F1+F2+F3+F4, com Harris):
           → pred OC se pelo menos 3 dos 4 filtros votam OC
      4. Eleger a lógica com maior accuracy na amostra de calibração
         (margem mínima de 0.1 pp para incluir F4 — só incluir se ajudar)

    Devolve dict de configuração completa para o parque (gravada no JSON).
    """
    cfg    = PARK_CFG[parque]
    labels = df_sub['label'].values

    print(f"\n  {parque}  ({len(df_sub)} ROIs):")
    thresholds, directions, accs_indiv = {}, {}, {}

    # ── Passo 1: calibrar limiar individual de cada feature ──────────────────
    for fi in ['f1', 'f2', 'f3', 'f4']:
        vals           = df_sub[fi].values
        t, d, a        = melhor_limiar(vals, labels)
        thresholds[fi] = t
        directions[fi] = d
        accs_indiv[fi] = round(a * 100, 3)
        lbl = cfg['f4_label'] if fi == 'f4' else cfg['feats'][int(fi[1]) - 1][2]
        print(f"    {fi} {lbl}: acc={a*100:.2f}%  t={t:.5f}  dir={d}")

    # ── Passo 2: comparar lógicas de votação ─────────────────────────────────
    def votos(feats_usar):
        # Contar quantos filtros votam OC para cada ROI
        total = np.zeros(len(labels), dtype=int)
        for fi in feats_usar:
            v = df_sub[fi].values
            t, d = thresholds[fi], directions[fi]
            total += (v > t).astype(int) if d == 'gt' else (v < t).astype(int)
        return total

    v3 = votos(['f1', 'f2', 'f3'])          # sem Harris (F4)
    v4 = votos(['f1', 'f2', 'f3', 'f4'])    # com Harris (F4)

    # Accuracy com cada lógica de maioria
    acc_2em3 = float(np.mean((v3 >= 2).astype(int) == labels))
    acc_3em4 = float(np.mean((v4 >= 3).astype(int) == labels))

    print(f"    Lógica 2-em-3 (sem F4): {acc_2em3*100:.2f}%")
    print(f"    Lógica 3-em-4 (com F4): {acc_3em4*100:.2f}%", end='')

    # ── Passo 3: eleger lógica ────────────────────────────────────────────────
    if acc_3em4 > acc_2em3 + 0.001:   # F4 melhora por margem clara → incluir
        logica, min_votos, feats_usar = 'maioria_3_em_4', 3, ['f1', 'f2', 'f3', 'f4']
        acc_eleita, usa_f4 = acc_3em4, True
        print(f"  ← eleita (+{(acc_3em4 - acc_2em3)*100:.2f} pp)")
    else:                              # F4 não ajuda → usar apenas 3 filtros
        logica, min_votos, feats_usar = 'maioria_2_em_3', 2, ['f1', 'f2', 'f3']
        acc_eleita, usa_f4 = acc_2em3, False
        print(f"  (F4 não melhora)")
        print(f"    → 2-em-3 eleita")

    # Acurácia por classe
    v_eleita = v4 if usa_f4 else v3
    for cls_label, cls_idx in [('livre', 0), ('ocupada', 1)]:
        mask    = labels == cls_idx
        acc_cls = float(np.mean((v_eleita[mask] >= min_votos) == (labels[mask] == 1)))
        n_cls   = mask.sum()
        print(f"      acc {cls_label:8s}: {acc_cls*100:.2f}%  ({n_cls} ROIs)")

    return {
        'parque'     : parque,
        'logica'     : logica,
        'min_votos'  : min_votos,
        'feats_usar' : feats_usar,
        'usa_f4'     : usa_f4,
        'thresholds' : thresholds,
        'directions' : directions,
        'acc_calib'  : round(acc_eleita * 100, 3),
        'accs_indiv' : {fi: round(a * 100, 3) for fi, a in accs_indiv.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — AVALIAÇÃO NO VALID (completo)
# ══════════════════════════════════════════════════════════════════════════════
def predizer_roi(feats_dict, park_calib):
    """Prediz ocupada (1) ou livre (0) para uma ROI dado o cfg do parque."""
    votos = 0
    for fi in park_calib['feats_usar']:
        t, d = park_calib['thresholds'][fi], park_calib['directions'][fi]
        v    = feats_dict.get(fi, 0.0)
        votos += int(v > t) if d == 'gt' else int(v < t)
    return 1 if votos >= park_calib['min_votos'] else 0


def avaliar_split(split, calib_por_parque):
    """
    Aplica os classificadores per-parque a TODAS as imagens do split.
    Devolve DataFrame com uma linha por vaga classificada.
    """
    data, anns_por_img, img_por_id = carregar_coco(split)
    img_dir  = os.path.join(DATASET, split)
    imgs     = data['images']
    n_imgs   = len(imgs)
    linhas   = []
    t0       = time.time()

    for i, img_info in enumerate(imgs):
        if (i + 1) % 250 == 0 or i == n_imgs - 1:
            el  = time.time() - t0
            eta = el / (i + 1) * (n_imgs - i - 1)
            print(f"  {split}: {i+1}/{n_imgs}  "
                  f"({el:.0f}s decorridos, ETA {eta:.0f}s)    ", end='\r')

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

            linhas.append({
                'split'     : split,
                'filename'  : img_info['file_name'],
                'parque'    : parque,
                'ann_id'    : ann['id'],
                'label_gt'  : 'livre'   if gt   == 0 else 'ocupada',
                'label_pred': 'livre'   if pred == 0 else 'ocupada',
                'correcto'  : int(pred == gt),
                'f1'        : round(feats.get('f1', 0.0), 6),
                'f2'        : round(feats.get('f2', 0.0), 6),
                'f3'        : round(feats.get('f3', 0.0), 6),
                'f4'        : round(feats.get('f4', 0.0), 6),
            })

    print()
    return pd.DataFrame(linhas)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 — MÉTRICAS E PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def calcular_metricas(df, nome_split):
    """Imprime accuracy global + por parque + precision/recall/F1."""
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

    print(f"\n{'─'*55}")
    print(f"  {nome_split.upper()}  —  {len(df):,} vagas classificadas")
    print(f"{'─'*55}")
    print(f"  Accuracy global : {acc:.2f}%")
    print(f"  Precision       : {prec:.2f}%  (previstas ocup. que são corretas)")
    print(f"  Recall          : {rec:.2f}%  (ocupadas reais detectadas)")
    print(f"  F1-score        : {f1:.2f}%")
    print(f"\n  Confusion matrix (real ↓ / pred →):")
    print(f"              livre    ocupada")
    print(f"  livre       {tn:>6,}   {fp:>7,}")
    print(f"  ocupada     {fn:>6,}   {tp:>7,}")

    print(f"\n  Accuracy por parque:")
    for park in PARQUES:
        sub = df[df['parque'] == park]
        if sub.empty:
            continue
        a  = sub['correcto'].mean() * 100
        n  = len(sub)
        ok = '✅' if a >= 80 else ('⚠️' if a >= 70 else '❌')
        print(f"    {park}: {a:.2f}%  ({n:,} vagas)  {ok}")

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def plot_confusion(m, split, out_path):
    """Heatmap da confusion matrix."""
    matrix = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    total  = matrix.sum()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    labels_rc = ['livre', 'ocupada']
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_rc)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_rc)
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
        f"Confusion Matrix — {split}  (per-parque)\n"
        f"Accuracy: {m['acc']:.2f}%   F1: {m['f1']:.2f}%",
        fontsize=10, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


def plot_acuracias(df, m_valid, calib_por_parque, out_path):
    """Bar chart com accuracy global + por parque + acc_calib."""
    parques  = PARQUES
    accs_v   = [df[df['parque'] == p]['correcto'].mean() * 100
                if not df[df['parque'] == p].empty else 0 for p in parques]
    accs_c   = [calib_por_parque[p]['acc_calib'] for p in parques]
    ns_v     = [len(df[df['parque'] == p]) for p in parques]

    labels_x = ['GLOBAL'] + parques
    accs_x   = [m_valid['acc']] + accs_v
    cores    = ['#2c3e50'] + [CORES_PARQUE[p] for p in parques]

    fig, ax = plt.subplots(figsize=(9, 6))
    x_pos   = np.arange(len(labels_x))
    bars    = ax.bar(x_pos, accs_x, color=cores, edgecolor='white',
                     linewidth=1.5, width=0.55, label='valid (completo)')

    # Pontinhos de calibração (amostra train)
    ax.scatter([1, 2, 3], accs_c, marker='D', s=80, color='black',
               zorder=5, label='calib (amostra train)')

    for bar, acc, n in zip(bars, accs_x, [len(df)] + ns_v):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{acc:.1f}%\n({n:,})", ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    for i, (p, ac) in enumerate(zip(parques, accs_c), 1):
        ax.text(i, ac - 1.5, f"{ac:.1f}%\n(calib)",
                ha='center', va='top', fontsize=7, color='black', alpha=0.75)

    ax.axhline(80, color='green',  linestyle='--', linewidth=1.2, alpha=0.55, label='mínimo 80%')
    ax.axhline(70, color='orange', linestyle='--', linewidth=1.2, alpha=0.55, label='mínimo 70%')
    ax.set_ylim(0, 112)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_x, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)

    # Lógica eleita por parque
    subtitles = []
    for p in parques:
        cp   = calib_por_parque[p]
        f4s  = ' +F4' if cp['usa_f4'] else ''
        subtitles.append(f"{p}: {cp['logica'].replace('_', ' ')}{f4s}")
    ax.set_title(
        f"Passo 3 v2 — Accuracy por parque  (split valid)\n"
        f"Per-parque: {' | '.join(subtitles)}",
        fontsize=10, fontweight='bold',
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


def plot_calib_summary(df_calib, calib_por_parque, out_path):
    """Violin plots de F1-F4 por parque com limiares calibrados."""
    fig, axes = plt.subplots(len(PARQUES), 4,
                             figsize=(16, len(PARQUES) * 3.5))
    fi_labels = ['f1', 'f2', 'f3', 'f4']

    for row, parque in enumerate(PARQUES):
        df_p      = df_calib[df_calib['parque'] == parque]
        cfg       = PARK_CFG[parque]
        cp        = calib_por_parque[parque]
        cor       = CORES_PARQUE[parque]
        vals_l    = df_p[df_p['label'] == 0]
        vals_o    = df_p[df_p['label'] == 1]

        for col, fi in enumerate(fi_labels):
            ax = axes[row, col]
            vl = vals_l[fi].values
            vo = vals_o[fi].values

            if len(vl) > 1 and len(vo) > 1:
                vp = ax.violinplot([vl, vo], positions=[1, 2],
                                   showmedians=True, showextrema=True)
                vp['bodies'][0].set_facecolor('#2ecc71')
                vp['bodies'][0].set_alpha(0.5)
                vp['bodies'][1].set_facecolor('#e74c3c')
                vp['bodies'][1].set_alpha(0.5)
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(2)

            t_opt = cp['thresholds'][fi]
            ax.axhline(t_opt, color='navy', linewidth=1.5, linestyle='--', alpha=0.8)

            if fi == 'f4':
                feat_lbl = cfg['f4_label']
                usada    = cp['usa_f4']
            else:
                feat_lbl = cfg['feats'][int(fi[1]) - 1][2]
                usada    = True
            acc_i = cp['accs_indiv'][fi]

            ax.set_xticks([1, 2])
            ax.set_xticklabels(['livre', 'ocup.'], fontsize=8)
            titulo = f"{feat_lbl}\nacc={acc_i:.1f}%"
            if fi == 'f4':
                titulo += '  ✅' if usada else '  ✗'
            ax.set_title(titulo, fontsize=7, fontweight='bold', color=cor)
            ax.grid(axis='y', alpha=0.3)
            if col == 0:
                ax.set_ylabel(
                    f"{parque}\n{cp['logica'].replace('_',' ')}",
                    fontsize=9, fontweight='bold', color=cor,
                )
            for sp in ax.spines.values():
                sp.set_edgecolor(cor)
                sp.set_linewidth(1.8)

    fig.suptitle(
        "Passo 3 v2 — Calibração per-parque  (amostra train)\n"
        "Verde=livre  Vermelho=ocupada  Azul tracejado=limiar  ✅/✗=F4 usada ou não",
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Guardado: {os.path.basename(out_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_total = time.time()

    # ── FASE 1: Calibração ───────────────────────────────────────────────────
    df_calib = extrair_calibracao()

    # ── FASE 2: Sweep de limiares por parque ─────────────────────────────────
    print(f"\nFASE 2 — Sweep de limiares ({N_SWEEP} pontos por feature)")
    print("  Compara 2-em-3 (sem F4) vs 3-em-4 (com F4) por parque.")

    calib_por_parque = {}
    for parque in PARQUES:
        df_sub = df_calib[df_calib['parque'] == parque].copy()
        calib_por_parque[parque] = calibrar_parque(df_sub, parque)

    # Guardar limiares em JSON
    limiares_json = {'versao': 'v2-per-parque'}
    for parque, cp in calib_por_parque.items():
        cfg = PARK_CFG[parque]
        limiares_json[parque] = {
            'logica'        : cp['logica'],
            'min_votos'     : cp['min_votos'],
            'usa_f4'        : cp['usa_f4'],
            'feats_usar'    : cp['feats_usar'],
            'features_desc' : {
                'f1': cfg['feats'][0][2],
                'f2': cfg['feats'][1][2],
                'f3': cfg['feats'][2][2],
                'f4': cfg['f4_label'],
            },
            'thresholds'    : cp['thresholds'],
            'directions'    : cp['directions'],
            'accs_indiv'    : cp['accs_indiv'],
            'acc_calib'     : cp['acc_calib'],
            'n_calib'       : int((df_calib['parque'] == parque).sum()),
        }

    out_json = os.path.join(_SCRIPT_DIR, 'passo3_limiares.json')
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(limiares_json, fh, indent=2, ensure_ascii=False)
    print(f"\n  Limiares guardados: {os.path.basename(out_json)}")

    # ── FASE 3: Avaliação no valid completo ──────────────────────────────────
    print(f"\nFASE 3 — Avaliação no valid completo")
    print("  (pode demorar 15-30 minutos conforme o hardware)\n")
    df_valid = avaliar_split('valid', calib_por_parque)

    out_csv = os.path.join(_SCRIPT_DIR, 'passo3_valid_predicoes.csv')
    df_valid.to_csv(out_csv, index=False)
    print(f"  CSV guardado: {os.path.basename(out_csv)}")

    # ── FASE 4: Métricas e plots ─────────────────────────────────────────────
    print(f"\nFASE 4 — Métricas e plots")
    m_valid = calcular_metricas(df_valid, 'VALID')

    out_conf = os.path.join(_SCRIPT_DIR, 'passo3_confusion.png')
    plot_confusion(m_valid, 'Valid', out_conf)

    out_bars = os.path.join(_SCRIPT_DIR, 'passo3_acuracias.png')
    plot_acuracias(df_valid, m_valid, calib_por_parque, out_bars)

    out_calib_plot = os.path.join(_SCRIPT_DIR, 'passo3_calib_summary.png')
    plot_calib_summary(df_calib, calib_por_parque, out_calib_plot)

    # ── Resumo final ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'═'*62}")
    print(f"  PASSO 3 v2 (per-parque) CONCLUÍDO  ({elapsed/60:.1f} min)")
    print(f"{'═'*62}")

    print(f"\n  Calibração por parque:")
    for parque, cp in calib_por_parque.items():
        f4_str = '+F4 ✅' if cp['usa_f4'] else 'sem F4'
        print(f"    {parque}: acc_calib={cp['acc_calib']:.2f}%  "
              f"lógica={cp['logica'].replace('_', ' ')} [{f4_str}]")

    print(f"\n  Resultados no valid completo:")
    print(f"    Accuracy global : {m_valid['acc']:.2f}%")
    print(f"    F1-score        : {m_valid['f1']:.2f}%")

    ok_global = m_valid['acc'] >= 80
    print(f"\n  Critério global (>80%): {'✅ PASSOU' if ok_global else '❌ NÃO PASSOU'}")

    parques_ok = []
    for park in PARQUES:
        sub = df_valid[df_valid['parque'] == park]
        if not sub.empty:
            a  = sub['correcto'].mean() * 100
            ok = a >= 70
            parques_ok.append(ok)
            status = '✅' if a >= 80 else '⚠️' if a >= 70 else '❌'
            print(f"  {park}: {a:.2f}% {status}")

    if all(parques_ok) and ok_global:
        print(f"\n  → PASSO 3 VALIDADO. Pode avançar para o Passo 4.")
    else:
        print(f"\n  → Rever limiares ou features antes de avançar.")

    print(f"\n  Ficheiros gerados:")
    for f in [out_json, out_csv, out_conf, out_bars, out_calib_plot]:
        print(f"    {os.path.basename(f)}")
