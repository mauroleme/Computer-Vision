# passo2_eda.py — versão 4
# Passo 2 — Análise exploratória de features com pipelines de pré-processamento
# Visão por Computador — MEEC / IPB 2025-2026
# Alunos: Caio Sant'Ana Oliveira (52963) - Mauro da Silva Leme (a52965)
#
# ─────────────────────────────────────────────────────────────────────────────
# PAPEL NO PROJECTO
# ─────────────────────────────────────────────────────────────────────────────
# Este é o SEGUNDO passo do pipeline. Corre uma vez antes do passo3.
#
# O objectivo é descobrir QUAIS features (descritores de textura/gradiente)
# melhor separam vagas Livres de Ocupadas, e em QUE pipeline de pré-
# processamento cada feature é mais discriminativa.
#
# Estratégia:
#   1. Amostrar N_POR_GRUPO ROIs livres e ocupadas de cada parque
#   2. Para cada feature × pipeline, calcular:
#      - Fisher ratio: mede a separabilidade entre as duas classes
#        (médias afastadas + variâncias baixas → Fisher alto → boa feature)
#      - Accuracy com threshold óptimo (melhor ponto de corte por F1-score)
#   3. Rankear features individualmente e em combinações de 2 e 3
#   4. Gravar CSV com todos os valores para o passo2b analisar por parque
#
# SAÍDAS GERADAS:
#   passo2_features.csv        → tabela de features por ROI (lida pelo passo2b)
#   passo2_top_indiv.png       → gráfico: top 15 features individuais (Fisher)
#   passo2_top_pares.png       → gráfico: top pares de features
#   passo2_top_triplas.png     → gráfico: top triplas de features
#   passo2_violins_top5.png    → distribuições das 5 melhores features
#
# COMO CORRER:
#   python passo2_eda.py
#   (demora ~5-15 min dependendo do hardware)
# ─────────────────────────────────────────────────────────────────────────────
#
# NOVIDADES v3 → v4:
#   + Feature nova : n_cantos (Harris corners count)
#   + 4 variantes de parâmetros × 12 pipelines = 48 novas colunas
#     Variantes: (thresh=0.01, min_dist=2) | (thresh=0.02, min_dist=3)
#               (thresh=0.05, min_dist=3) | (thresh=0.10, min_dist=5)
#   Justificação: passo2_hough_corners_test.py mostrou Fisher>1.0 nos 3
#                 parques para n_cantos com thresh=0.02/min_dist=3.
#                 Vagas ocupadas têm mais cantos (janelas, bordas do carro).
#                 Testamos os 4 limiares para encontrar o mais discriminativo.
#   Total: 12 pipelines × 17 features = 204 + HSV = 205 features
#
# NOVIDADES v2 → v3:
#   + Pipelines novos: median, unsharp, gauss_median, median_gauss,
#                      tophat_gauss, unsharp_gauss, gauss_unsharp
#   + Features novas : laplacian_mean, prewitt_mean, p90_p10, hog_std
#   Removidos: clahe_gauss, clahe_tophat (confirmados maus na v2)
#
# PIPELINES (12):
#   raw           → sem pré-processamento (referência)
#   gauss         → Gaussiano σ=1 (suavização — melhor da v2)
#   median        → mediana 3×3 (remove ruído impulsivo, preserva bordas)
#   tophat        → white tophat disk(5)
#   unsharp       → unsharp mask r=2, a=1 (realça bordas — inverso do Gaussiano)
#   clahe         → CLAHE (controlo — esperado fraco, confirmado v2)
#   gauss_tophat  → Gaussiano → tophat
#   gauss_median  → Gaussiano → mediana (suavizar → remover ruído residual)
#   median_gauss  → mediana → Gaussiano (desruído forte → suavização leve)
#   tophat_gauss  → tophat → Gaussiano  (detectar estruturas → suavizar)
#   unsharp_gauss → unsharp → Gaussiano (realçar bordas → suavizar moderadamente)
#   gauss_unsharp → Gaussiano → unsharp (suavizar → realçar bordas residuais)

import json
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter as scipy_median
from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import sobel, gaussian, laplace, prewitt, unsharp_mask
from skimage.exposure import equalize_adapthist
from skimage.feature import (canny, graycomatrix, graycoprops,
                             local_binary_pattern, corner_harris, corner_peaks)
from skimage.feature import hog as sk_hog
from skimage.morphology import white_tophat, disk
from skimage.measure import shannon_entropy

# ── Configuração ─────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET      = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot'))
JSON_PATH    = os.path.join(DATASET, 'train', '_annotations.coco.json')
IMG_DIR      = os.path.join(DATASET, 'train')

N_POR_GRUPO  = 200    # 100 livres + 100 ocupadas por parque (mesmo que v2)
CAT_LIVRE    = 1
CAT_OCUPADA  = 2
SE3          = disk(3)
SE5          = disk(5)
SEMENTE      = 42
TOP_INDIV    = 15
TOP_TRIPLAS  = 8

# Harris corners — 4 variantes de parâmetros (v4)
# Cada variante gera features: {pipeline}_n_cantos_t{thresh*100:02d}_d{min_dist}
# thresh baixo → mais cantos detectados (mais sensível)
# min_dist alto → cantos mais espaçados (menos redundância)
CORNER_PARAMS = [
    (0.01, 2),   # muito sensível  — capta bordas suaves e cantos finos
    (0.02, 3),   # moderado        — igual ao passo2_hough_corners_test.py (Fisher>1.0)
    (0.05, 3),   # selectivo       — apenas cantos com resposta Harris forte
    (0.10, 5),   # muito selectivo — só cantos muito nítidos e espaçados
]

# ── Pipelines (12) ───────────────────────────────────────────────────
# Aplicados à IMAGEM COMPLETA antes de cortar ROIs.
PIPELINES = {
    'raw'          : lambda g: g,
    'gauss'        : lambda g: gaussian(g, sigma=1),
    'median'       : lambda g: scipy_median(g, size=3),
    'tophat'       : lambda g: white_tophat(g, SE5),
    'unsharp'      : lambda g: np.clip(unsharp_mask(g, radius=2, amount=1.0), 0.0, 1.0),
    'clahe'        : lambda g: equalize_adapthist(g, clip_limit=0.03),
    'gauss_tophat' : lambda g: white_tophat(gaussian(g, sigma=1), SE5),
    'gauss_median' : lambda g: scipy_median(gaussian(g, sigma=1), size=3),
    'median_gauss' : lambda g: gaussian(scipy_median(g, size=3), sigma=1),
    'tophat_gauss' : lambda g: gaussian(white_tophat(g, SE5), sigma=1),
    'unsharp_gauss': lambda g: gaussian(
                         np.clip(unsharp_mask(g, radius=2, amount=1.0), 0.0, 1.0), sigma=1),
    'gauss_unsharp': lambda g: np.clip(
                         unsharp_mask(gaussian(g, sigma=1), radius=2, amount=1.0), 0.0, 1.0),
}

PIPELINE_NAMES = list(PIPELINES.keys())

FEATURES_BASE = [
    'sobel_mean',        # bordas Sobel (1ª derivada)
    'canny_density',     # fracção de bordas Canny
    'entropy',           # entropia Shannon
    'std_intensity',     # desvio-padrão de intensidade
    'dark_ratio',        # fracção de píxeis escuros
    'glcm_contrast',     # contraste GLCM
    'glcm_homogeneity',  # homogeneidade GLCM
    'tophat_mean',       # white tophat (estruturas claras finas)
    'lbp_std',           # LBP std (textura local)
    'laplacian_mean',    # bordas pela 2ª derivada (Laplaciano)
    'prewitt_mean',      # bordas Prewitt (sensibilidade direcional diferente de Sobel)
    'p90_p10',           # alcance dinâmico — percentil 90 menos percentil 10
    'hog_std',           # HOG std — variabilidade dos gradientes orientados
    # v4 — Harris corners: 4 variantes de (thresh, min_dist)
    # nome: n_cantos_t{thresh*100:02d}_d{min_dist}
    'n_cantos_t01_d2',   # muito sensível  (thresh=0.01, min_dist=2)
    'n_cantos_t02_d3',   # moderado        (thresh=0.02, min_dist=3) ← Fisher>1.0 confirmado
    'n_cantos_t05_d3',   # selectivo       (thresh=0.05, min_dist=3)
    'n_cantos_t10_d5',   # muito selectivo (thresh=0.10, min_dist=5)
]


def parse_feat_name(feat):
    """Devolve (pipeline, feature_base) para qualquer nome de feature composto.
    Usa lista de pipelines conhecidos, do mais longo para o mais curto,
    para evitar matches errados em nomes compostos como gauss_tophat vs gauss.
    """
    for pp in sorted(PIPELINE_NAMES, key=len, reverse=True):
        if feat.startswith(pp + '_'):
            return pp, feat[len(pp) + 1:]
    return '', feat


# ── Extracção de 13 features de uma ROI processada ───────────────────
def extrair_features_roi(roi_proc, med_global_proc):
    """
    roi_proc        : ROI cortada da imagem pré-processada (float [0,1])
    med_global_proc : mediana da imagem completa pós-processamento
    Devolve dicionário com 13 features, ou None se a ROI for inválida.
    """
    if roi_proc.size < 16:
        return None

    roi_proc = roi_proc.astype(float)
    uint8_p  = (roi_proc * 255).astype(np.uint8)   # LBP precisa de inteiros
    q64_p    = (roi_proc *  63).astype(np.uint8)   # GLCM: valores < levels=64

    # 1. Sobel (1ª derivada, magnitude)
    f_sobel = float(np.mean(sobel(roi_proc)))

    # 2. Canny density
    f_canny = float(np.mean(canny(roi_proc, sigma=1.5)))

    # 3. Shannon entropy
    f_entr  = float(shannon_entropy(roi_proc))

    # 4. Std intensity
    f_std   = float(np.std(roi_proc))

    # 5. Dark ratio (adaptativo à mediana global da imagem processada)
    limiar  = max(0.02, med_global_proc - 0.10)
    f_dark  = float(np.mean(roi_proc < limiar))

    # 6 & 7. GLCM
    glcm    = graycomatrix(q64_p, distances=[1], angles=[0],
                           levels=64, symmetric=True, normed=True)
    f_cont  = float(graycoprops(glcm, 'contrast')[0, 0])
    f_homog = float(graycoprops(glcm, 'homogeneity')[0, 0])

    # 8. White tophat sobre a ROI já processada
    f_top   = float(np.mean(white_tophat(roi_proc, SE3)))

    # 9. LBP std
    lbp     = local_binary_pattern(uint8_p, P=8, R=1, method='uniform')
    f_lbp   = float(np.std(lbp))

    # 10. Laplaciano (2ª derivada) — valor absoluto médio
    f_lap   = float(np.mean(np.abs(laplace(roi_proc))))

    # 11. Prewitt (magnitude do gradiente, sensibilidade direcional diferente de Sobel)
    f_prew  = float(np.mean(prewitt(roi_proc)))

    # 12. Alcance dinâmico (p90 - p10) — menos sensível a outliers que std
    f_pct   = float(np.percentile(roi_proc, 90) - np.percentile(roi_proc, 10))

    # 13. HOG std — variabilidade dos gradientes orientados
    try:
        hog_desc = sk_hog(roi_proc, orientations=8,
                          pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                          feature_vector=True)
        f_hog = float(np.std(hog_desc)) if hog_desc.size > 0 else 0.0
    except Exception:
        f_hog = 0.0

    # 14. Harris corners count — 4 variantes de parâmetros (v4)
    # corner_harris calculado UMA vez; corner_peaks reutiliza a resposta com
    # diferentes limiares e distâncias mínimas → eficiente
    h_response = corner_harris(roi_proc)
    cantos_feats = {}
    for thresh, min_d in CORNER_PARAMS:
        try:
            pts = corner_peaks(h_response,
                               min_distance=min_d,
                               threshold_rel=thresh)
            cnt = float(len(pts))
        except Exception:
            cnt = 0.0
        tag = f'n_cantos_t{int(thresh * 100):02d}_d{min_d}'
        cantos_feats[tag] = cnt

    result = {
        'sobel_mean'      : f_sobel,
        'canny_density'   : f_canny,
        'entropy'         : f_entr,
        'std_intensity'   : f_std,
        'dark_ratio'      : f_dark,
        'glcm_contrast'   : f_cont,
        'glcm_homogeneity': f_homog,
        'tophat_mean'     : f_top,
        'lbp_std'         : f_lbp,
        'laplacian_mean'  : f_lap,
        'prewitt_mean'    : f_prew,
        'p90_p10'         : f_pct,
        'hog_std'         : f_hog,
    }
    result.update(cantos_feats)
    return result


# ── Carregar COCO JSON ────────────────────────────────────────────────
print("A carregar COCO JSON...")
with open(JSON_PATH, encoding='utf-8') as f:
    data = json.load(f)

anns_por_img = {}
for ann in data['annotations']:
    anns_por_img.setdefault(ann['image_id'], []).append(ann)

img_por_id = {img['id']: img for img in data['images']}

grupos_ids = {28: [], 40: [], 100: []}
for img in data['images']:
    n = len(anns_por_img.get(img['id'], []))
    if n in grupos_ids:
        grupos_ids[n].append(img['id'])

print(f"  G28:{len(grupos_ids[28])}  G40:{len(grupos_ids[40])}  G100:{len(grupos_ids[100])} imagens")


# ── Amostragem e extracção por grupo ─────────────────────────────────
def extrair_grupo(n_vagas, n_amostras=N_POR_GRUPO):
    rng = np.random.default_rng(SEMENTE)
    img_ids = list(grupos_ids[n_vagas])
    rng.shuffle(img_ids)

    pool_l, pool_o = [], []
    for iid in img_ids:
        for ann in anns_por_img.get(iid, []):
            (pool_l if ann['category_id'] == CAT_LIVRE else pool_o).append((iid, ann))
        if len(pool_l) >= n_amostras and len(pool_o) >= n_amostras:
            break

    n_each = min(n_amostras // 2, len(pool_l), len(pool_o))
    idx_l  = rng.choice(len(pool_l), n_each, replace=False)
    idx_o  = rng.choice(len(pool_o), n_each, replace=False)
    sel    = [pool_l[i] for i in idx_l] + [pool_o[i] for i in idx_o]

    por_imagem = {}
    for iid, ann in sel:
        por_imagem.setdefault(iid, []).append(ann)

    linhas = []
    n_imgs = 0

    for iid, anns in por_imagem.items():
        img_info = img_por_id[iid]
        img_path = os.path.join(IMG_DIR, img_info['file_name'])
        image    = imread(img_path)
        gray_raw = rgb2gray(image)
        hsv_orig = rgb2hsv(image)
        H, W     = gray_raw.shape

        # Pré-processar imagem completa — uma vez por imagem, para todos os pipelines
        pp_imgs = {}
        pp_meds = {}
        for pp_nome, pp_fn in PIPELINES.items():
            gp               = pp_fn(gray_raw)
            pp_imgs[pp_nome] = gp
            pp_meds[pp_nome] = float(np.median(gp))

        for ann in anns:
            x, y, w, h = ann['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = min(int(x + w), W), min(int(y + h), H)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            label = 'livre' if ann['category_id'] == CAT_LIVRE else 'ocupada'
            linha = {
                'parque'   : f'G{n_vagas}',
                'image_id' : iid,
                'label_gt' : label,
                'label_num': 0 if label == 'livre' else 1,
            }

            for pp_nome, gp in pp_imgs.items():
                roi   = gp[y1:y2, x1:x2]
                feats = extrair_features_roi(roi, pp_meds[pp_nome])
                if feats is None:
                    continue
                for feat_nome, val in feats.items():
                    linha[f'{pp_nome}_{feat_nome}'] = val

            roi_hsv = hsv_orig[y1:y2, x1:x2]
            linha['hsv_saturation'] = float(np.mean(roi_hsv[:, :, 1]))
            linhas.append(linha)

        n_imgs += 1

    print(f"  G{n_vagas}: {len(linhas)} ROIs de {n_imgs} imagens")
    return linhas


n_feat_total = len(PIPELINES) * len(FEATURES_BASE) + 1  # +1 HSV
print(f"\nA extrair features ({len(PIPELINES)} pipelines × {len(FEATURES_BASE)} features + HSV = {n_feat_total} total)...")
print(f"  Novidade v4: {len(CORNER_PARAMS)} variantes de Harris corners × {len(PIPELINES)} pipelines = {len(CORNER_PARAMS)*len(PIPELINES)} novas colunas")
print("Pode demorar 25-45 minutos...\n")

linhas = []
for n in [28, 40, 100]:
    linhas.extend(extrair_grupo(n))

df = pd.DataFrame(linhas)

meta_cols = ['parque', 'image_id', 'label_gt', 'label_num']
feat_cols  = [c for c in df.columns if c not in meta_cols]
df[feat_cols] = df[feat_cols].fillna(0)

print(f"\nDataFrame: {len(df)} ROIs × {len(feat_cols)} features")
print(df.groupby(['parque', 'label_gt']).size().to_string())

csv_feat = os.path.join(_SCRIPT_DIR, 'passo2_features.csv')
df.to_csv(csv_feat, index=False)
print(f"\nGuardado: {csv_feat}")


# ── Funções de análise ────────────────────────────────────────────────
def fisher_ratio(s_livre, s_ocup):
    denom = s_livre.var() + s_ocup.var()
    if denom < 1e-12:
        return 0.0
    return float((s_livre.mean() - s_ocup.mean()) ** 2 / denom)


def melhor_limiar(vals, labels):
    """Sweep 200 limiares; devolve (threshold, direction, accuracy)."""
    ts   = np.linspace(vals.min(), vals.max(), 200)
    best = (0.0, 'gt', 0.0)
    for t in ts:
        for direc, pred in [('gt', (vals > t).astype(int)),
                             ('lt', (vals < t).astype(int))]:
            acc = float(np.mean(pred == labels))
            if acc > best[2]:
                best = (t, direc, acc)
    return best


def predizer(df_sub, feats_info, logica):
    """
    feats_info : lista de (col_name, threshold, direction)
    logica     : 'and'      → ocupada se QUALQUER feature diz ocupada
                 'majority' → ocupada se MAIORIA diz ocupada
    """
    labels     = df_sub['label_num'].values
    n          = len(feats_info)
    votos_ocup = np.zeros(len(labels), dtype=int)
    for feat, t, direc in feats_info:
        v = df_sub[feat].values
        votos_ocup += ((v > t) if direc == 'gt' else (v < t)).astype(int)
    if logica == 'and':
        pred = (votos_ocup >= 1).astype(int)
    else:
        thresh = n // 2 + 1
        pred   = (votos_ocup >= thresh).astype(int)
    return float(np.mean(pred == labels))


# ── Fase 1: features individuais ─────────────────────────────────────
print("\n── Fase 1: features individuais ──────────────────────────")
info_feat  = {}
resultados = []

for feat in feat_cols:
    vals   = df[feat].values
    labels = df['label_num'].values
    t, direc, acc = melhor_limiar(vals, labels)
    fish  = fisher_ratio(df[df['label_gt'] == 'livre'][feat],
                         df[df['label_gt'] == 'ocupada'][feat])
    info_feat[feat] = (t, direc)
    resultados.append({
        'n_features': 1, 'logica': 'single',
        'combinacao': feat,
        'accuracy'  : round(acc * 100, 2),
        'fisher'    : round(fish, 4),
    })

df_res1 = pd.DataFrame(resultados).sort_values('accuracy', ascending=False)
print("\nTOP 20 features individuais:")
print(df_res1.head(20)[['combinacao', 'accuracy', 'fisher']].to_string(index=False))

top_indiv = df_res1.head(TOP_INDIV)['combinacao'].tolist()


# ── Fase 2: pares das top 15 ─────────────────────────────────────────
pares = list(itertools.combinations(top_indiv, 2))
print(f"\n── Fase 2: {len(pares)} pares × 2 lógicas ─────────────────")

for f1, f2 in pares:
    fi = [(f1, *info_feat[f1]), (f2, *info_feat[f2])]
    for logica in ('and', 'majority'):
        acc  = predizer(df, fi, logica)
        fish = np.mean([
            fisher_ratio(df[df['label_gt']=='livre'][f1], df[df['label_gt']=='ocupada'][f1]),
            fisher_ratio(df[df['label_gt']=='livre'][f2], df[df['label_gt']=='ocupada'][f2]),
        ])
        resultados.append({
            'n_features': 2, 'logica': logica,
            'combinacao': f'{f1}  +  {f2}',
            'accuracy'  : round(acc * 100, 2),
            'fisher'    : round(fish, 4),
        })

df_res2 = pd.DataFrame(resultados).sort_values('accuracy', ascending=False)

top_pares_feats = []
for _, row in df_res2[df_res2['n_features'] == 2].head(10).iterrows():
    for p in row['combinacao'].split('+'):
        top_pares_feats.append(p.strip())
pool_triplas = list(dict.fromkeys(top_indiv[:5] + top_pares_feats))[:TOP_TRIPLAS]


# ── Fase 3: triplas das top 8 ────────────────────────────────────────
triplas = list(itertools.combinations(pool_triplas, 3))
print(f"\n── Fase 3: {len(triplas)} triplas × 2 lógicas ──────────────")

for f1, f2, f3 in triplas:
    fi = [(f1, *info_feat[f1]), (f2, *info_feat[f2]), (f3, *info_feat[f3])]
    for logica in ('and', 'majority'):
        acc  = predizer(df, fi, logica)
        fish = np.mean([
            fisher_ratio(df[df['label_gt']=='livre'][f],
                         df[df['label_gt']=='ocupada'][f])
            for f in [f1, f2, f3]
        ])
        resultados.append({
            'n_features': 3, 'logica': logica,
            'combinacao': f'{f1}  +  {f2}  +  {f3}',
            'accuracy'  : round(acc * 100, 2),
            'fisher'    : round(fish, 4),
        })


# ── Tabela final ──────────────────────────────────────────────────────
df_res = (pd.DataFrame(resultados)
            .sort_values(['accuracy', 'n_features'], ascending=[False, True])
            .reset_index(drop=True))

csv_res = os.path.join(_SCRIPT_DIR, 'passo2_resultados.csv')
df_res.to_csv(csv_res, index=False)
print(f"\nGuardado: {csv_res}")

print("\n=== TOP 20 COMBINAÇÕES (pipeline + feature + lógica) ===")
print(df_res.head(20)[['n_features', 'logica', 'combinacao', 'accuracy']].to_string(index=False))


# ── Violin plots — top 10 features individuais ────────────────────────
print("\nA gerar violin plots...")
top10 = df_res[df_res['n_features'] == 1].head(10)['combinacao'].tolist()

fig, axes = plt.subplots(2, 5, figsize=(24, 9))
axes  = axes.flatten()
marcas = {'G28': 'o', 'G40': 's', 'G100': '^'}

for idx, feat in enumerate(top10):
    ax = axes[idx]
    dados_l = df[df['label_gt'] == 'livre'  ][feat].values
    dados_o = df[df['label_gt'] == 'ocupada'][feat].values

    vp = ax.violinplot([dados_l, dados_o], positions=[1, 2],
                       showmedians=True, showextrema=True)
    for body, cor in zip(vp['bodies'], ['#2ecc71', '#e74c3c']):
        body.set_facecolor(cor)
        body.set_alpha(0.5)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(2)

    rng = np.random.default_rng(SEMENTE)
    for parque, marca in marcas.items():
        for pos, classe, cor in [(1, 'livre', '#27ae60'), (2, 'ocupada', '#c0392b')]:
            v = df[(df['parque'] == parque) & (df['label_gt'] == classe)][feat].values
            j = rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(np.full(len(v), pos) + j, v,
                       s=6, alpha=0.4, marker=marca, color=cor)

    t, direc, acc_i = melhor_limiar(df[feat].values, df['label_num'].values)
    fish = fisher_ratio(df[df['label_gt'] == 'livre'][feat],
                        df[df['label_gt'] == 'ocupada'][feat])
    ax.axhline(t, color='navy', linewidth=1.5, linestyle='--')

    pp_label, feat_label = parse_feat_name(feat)
    ax.set_title(
        f"[{pp_label}]  {feat_label}\n"
        f"acc={acc_i*100:.1f}%   fisher={fish:.3f}",
        fontsize=8, fontweight='bold'
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['livre', 'ocupada'], fontsize=9)
    ax.grid(axis='y', alpha=0.3)

handles = [
    plt.Line2D([0],[0], marker='o', color='gray', ls='None', ms=7, label='G28'),
    plt.Line2D([0],[0], marker='s', color='gray', ls='None', ms=7, label='G40'),
    plt.Line2D([0],[0], marker='^', color='gray', ls='None', ms=7, label='G100'),
    plt.Line2D([0],[0], color='navy', ls='--', lw=1.5, label='limiar óptimo'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
fig.suptitle(
    f"Passo 2 v4 — Top 10 features individuais  |  [pipeline] feature\n"
    f"12 pipelines × 17 features + HSV = {len(feat_cols)} features testadas   ·   600 ROIs do split train",
    fontsize=12, fontweight='bold', y=1.01
)
plt.tight_layout(rect=[0, 0.05, 1, 1])

out_v = os.path.join(_SCRIPT_DIR, 'passo2_violins.png')
plt.savefig(out_v, dpi=150, bbox_inches='tight')
print(f"Guardado: {out_v}")
plt.close()


# ── Ranking top 25 combinações ────────────────────────────────────────
print("A gerar gráfico de ranking...")
top25 = df_res.head(25).copy()

def formatar_label(row):
    partes = [p.strip() for p in row['combinacao'].split('+')]
    def fmt(s):
        pp, fb = parse_feat_name(s.strip())
        return f"[{pp}] {fb}" if pp else s.strip()
    txt = '\n+ '.join(fmt(p) for p in partes)
    if row['n_features'] > 1:
        txt += f"  ({row['logica']})"
    return txt

top25['label'] = top25.apply(formatar_label, axis=1)
cores_n = {1: '#3498db', 2: '#e67e22', 3: '#9b59b6'}

fig2, ax2 = plt.subplots(figsize=(15, 12))
ax2.barh(range(len(top25)), top25['accuracy'],
         color=[cores_n[n] for n in top25['n_features']],
         edgecolor='white', linewidth=0.5)

for i, (_, row) in enumerate(top25.iterrows()):
    ax2.text(row['accuracy'] + 0.2, i,
             f"{row['accuracy']:.1f}%",
             va='center', fontsize=8, fontweight='bold')

ax2.set_yticks(range(len(top25)))
ax2.set_yticklabels(top25['label'], fontsize=7.5)
ax2.invert_yaxis()
ax2.set_xlabel('Accuracy (%)', fontsize=11)
ax2.set_xlim(0, 110)
ax2.set_title(
    "Passo 2 v4 — Ranking top 25 combinações\n"
    "v4: Harris corners (4 limiares × 12 pipelines = 48 novas features)",
    fontsize=12, fontweight='bold'
)
ax2.grid(axis='x', alpha=0.3)

legend_h = [
    plt.Rectangle((0,0), 1, 1, color=cores_n[1], label='1 feature'),
    plt.Rectangle((0,0), 1, 1, color=cores_n[2], label='2 features'),
    plt.Rectangle((0,0), 1, 1, color=cores_n[3], label='3 features'),
]
ax2.legend(handles=legend_h, loc='lower right', fontsize=10)
plt.tight_layout()

out_r = os.path.join(_SCRIPT_DIR, 'passo2_ranking.png')
plt.savefig(out_r, dpi=150, bbox_inches='tight')
print(f"Guardado: {out_r}")
plt.close()


# ── Resumo final ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("PASSO 2 v3 CONCLUÍDO")
print("="*60)
best = df_res.iloc[0]
print(f"\nMelhor combinação encontrada:")
print(f"  Features : {best['combinacao']}")
print(f"  Lógica   : {best['logica']}")
print(f"  Accuracy : {best['accuracy']}%")

# Comparação com versões anteriores
print(f"\nBenchmark v2: 90.5%  (gauss_sobel_mean + gauss_glcm_contrast + raw_std_intensity)")
print(f"Benchmark v3: 91.0%  (median_gauss_prewitt_mean + gauss_median_glcm_contrast + ...)")
delta = best['accuracy'] - 91.0
sinal = '+' if delta >= 0 else ''
print(f"Variação v4:  {sinal}{delta:.1f} pp vs v3  ({'melhoria' if delta > 0.5 else 'plateau' if abs(delta) <= 0.5 else 'regressão'})")

print(f"\nTop 5 features individuais:")
top5_ind = df_res[df_res['n_features'] == 1].head(5)
for _, r in top5_ind.iterrows():
    pp, fb = parse_feat_name(r['combinacao'])
    print(f"  [{pp}] {fb:25s}  acc={r['accuracy']:.1f}%  fisher={r['fisher']:.3f}")

print(f"\nFicheiros gerados:")
for f in [csv_feat, csv_res, out_v, out_r]:
    print(f"  {os.path.basename(f)}")
print(f"\nPróximo passo: usar estas features no classificador (Passo 3)")
print(f"\nNota v4: se n_cantos_* aparecer no top 5 → inclui no Passo 3 como F4")
print(f"         se não aparecer → descarta e usa as 3 features da v3")
