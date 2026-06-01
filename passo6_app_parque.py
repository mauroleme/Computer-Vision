# passo6_app_parque.py
# Interface Tkinter -- Classificador de Vagas de Parque
# Passo 6 -- Visao por Computador -- MEEC / IPB 2025-2026
# Alunos: Caio Sant'Ana Oliveira (52963) - Mauro da Silva Leme (a52965)
#
# ─────────────────────────────────────────────────────────────────────────────
# PAPEL NO PROJECTO — APLICACAO PRINCIPAL (DEMO AO PROFESSOR)
# ─────────────────────────────────────────────────────────────────────────────
# Este é o SEXTO e último passo — a aplicacao interactiva que integra tudo:
#   • Carrega os limiares calibrados (passo3_limiares.json)
#   • Carrega as anotacoes COCO dos 3 parques (G28 / G40 / G100)
#   • Permite abrir qualquer imagem do dataset e classificar as vagas
#   • Mostra o resultado visualmente com overlay verde/vermelho
#   • Permite inspecionar vaga a vaga: features, votos e decisao
#   • Abre visualizador de pipeline detalhado para cada vaga (passo7_pipeline_viz)
#
# COMO USAR (demo ao professor):
#   1. python passo6_app_parque.py
#   2. Clicar "Abrir Imagem" → escolher qualquer .jpg do dataset
#   3. Clicar "Classificar" → aguardar overlay verde/vermelho
#   4. Clicar numa vaga → ver features e votos no painel direito
#   5. Clicar "🔍 Como foi avaliado?" → janela com o pipeline completo
#   6. Clicar "🗺 Como mapear vagas?" → Canny adaptativo interactivo (demo de mapeamento)
#
# MOTOR DE CLASSIFICACAO (funcoes processar_img, extrair_feats, predizer):
#   Identico ao passo3 e passo4 — os mesmos algoritmos, os mesmos limiares.
#   A diferenca e apenas a interface grafica em vez de um loop de batch.
# ─────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import threading
import subprocess
import sys
import tempfile
import json
import os
import time
import warnings
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, gaussian, unsharp_mask
from skimage.feature import graycomatrix, graycoprops, corner_harris, corner_peaks
from scipy.ndimage import median_filter as scipy_median

warnings.filterwarnings('ignore')

# Pipeline viz — lançado como subprocess separado para evitar conflito TkAgg + thread
_PIPELINE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'passo7_pipeline_viz.py')
_PIPELINE_OK = os.path.exists(_PIPELINE_SCRIPT)

# Canny interactivo — demonstração de como as vagas podem ser mapeadas
_CANNY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'passo_teste_canny_interativo.py')
_CANNY_OK = os.path.exists(_CANNY_SCRIPT)

# ── Caminhos ────────────────────────────────────────────────────────────────────
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET          = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot'))
DATASET_G100     = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot_g100_crop'))
CALIB_FILE       = os.path.join(_SCRIPT_DIR, 'passo3_limiares.json')
EXTRA_VAGAS_FILE = os.path.join(_SCRIPT_DIR, 'vagas_indefinidas', 'vagas_extra_g100_crop.json')

CAT_LIVRE   = 1
CAT_OCUPADA = 2

# ── Tema de cores (Catppuccin Mocha) ────────────────────────────────────────────
BG        = '#1e1e2e'
BG_PANEL  = '#181825'
BG_CANVAS = '#11111b'
BG_TB     = '#12121c'
FG        = '#cdd6f4'
FG_DIM    = '#6c7086'
ACCENT    = '#89b4fa'
OK_G      = '#a6e3a1'
ERR_R     = '#f38ba8'
WARN_O    = '#fab387'
INFO_B    = '#89dceb'
SEP       = '#313244'

# Overlay RGBA (fill dos rectangulos em PIL)
# Apenas 2 cores: verde=livre, vermelho=ocupada  (independente de acerto)
COL_LIVRE = (  0, 210,  80, 110)
COL_OCUP  = (220,  40,  40, 120)

# Bordas dos rectangulos (RGB)
BRD_LIVRE = (  0, 230,  80)
BRD_OCUP  = (230,  40,  40)


# ── Pipelines e features (identicos ao passo4_avaliacao_final.py) ───────────────
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

PARK_CFG = {
    'G28': {
        'n_vagas'    : 28,
        'feats'      : [('unsharp',      'std_intensity',    'std_intensity [unsharp]'),
                        ('gauss',        'glcm_contrast',    'glcm_contrast [gauss]'),
                        ('raw',          'std_intensity',    'std_intensity [raw]')],
        'f4_pipeline': 'unsharp', 'f4_thresh': 0.05, 'f4_min_dist': 3,
        'cor'        : '#2980b9',
    },
    'G40': {
        'n_vagas'    : 40,
        'feats'      : [('median_gauss',  'p90_p10',       'p90_p10 [median_gauss]'),
                        ('unsharp',       'dark_ratio',    'dark_ratio [unsharp]'),
                        ('unsharp_gauss', 'std_intensity', 'std_intensity [unsharp_gauss]')],
        'f4_pipeline': 'gauss_unsharp', 'f4_thresh': 0.01, 'f4_min_dist': 2,
        'cor'        : '#27ae60',
    },
    'G100': {
        'n_vagas'    : 100,
        'feats'      : [('median_gauss', 'glcm_homogeneity', 'glcm_homog [median_gauss]'),
                        ('gauss_median', 'sobel_mean',       'sobel_mean [gauss_median]'),
                        ('median_gauss', 'prewitt_mean',     'prewitt_mean [median_gauss]')],
        'f4_pipeline': 'median', 'f4_thresh': 0.05, 'f4_min_dist': 3,
        'cor'        : '#e67e22',
    },
}
N_VAGAS_TO_PARQUE = {v['n_vagas']: p for p, v in PARK_CFG.items()}


def _glcm(roi, prop):
    q = (np.clip(roi, 0.0, 1.0) * 63).astype(np.uint8)
    g = graycomatrix(q, [1], [0], levels=64, symmetric=True, normed=True)
    return float(graycoprops(g, prop)[0, 0])

FEAT_FUNCS = {
    'std_intensity'   : lambda roi, **_: float(np.std(roi)),
    'p90_p10'         : lambda roi, **_: float(
        np.percentile(roi, 90) - np.percentile(roi, 10)),
    'dark_ratio'      : lambda roi, med_global=None, **_: float(
        np.mean(roi < max(0.02, (med_global or 0.5) - 0.10))),
    'glcm_contrast'   : lambda roi, **_: _glcm(roi, 'contrast'),
    'glcm_homogeneity': lambda roi, **_: _glcm(roi, 'homogeneity'),
    'sobel_mean'      : lambda roi, **_: float(np.mean(sobel(roi))),
    'prewitt_mean'    : lambda roi, **_: float(np.mean(prewitt(roi))),
}

def n_cantos(roi, thresh, min_dist):
    try:
        return float(len(corner_peaks(corner_harris(roi),
                                      min_distance=min_dist, threshold_rel=thresh)))
    except Exception:
        return 0.0

def processar_img(gray, parque):
    """
    Pré-processa a imagem completa aplicando os pipelines necessários para este parque.

    Em vez de processar a imagem pixel-a-pixel por cada vaga, aplica os filtros
    UMA VEZ à imagem inteira e guarda as versões processadas em memória.
    Quando extrair_feats() precisar da ROI filtrada, já está pronta — é só recortar.
    Isto é muito mais rápido do que re-filtrar 100 vezes (uma por vaga).
    """
    cfg  = PARK_CFG[parque]
    # Identificar quais pipelines este parque precisa (evitar calcular os desnecessários)
    pips = set(pp for pp, _, _ in cfg['feats']) | {cfg['f4_pipeline']}
    pp   = {p: PIPELINE_FUNCS[p](gray) for p in pips}
    # dark_ratio precisa da mediana global da imagem como referência de luminosidade
    med_g = {}
    for p, feat, _ in cfg['feats']:
        if feat == 'dark_ratio' and p not in med_g:
            med_g[p] = float(np.median(pp[p]))
    return pp, med_g

def extrair_feats(pp, med_g, parque, ann, H, W):
    """
    Extrai os 4 valores de feature (f1, f2, f3, f4) para uma única vaga.

    Recebe as imagens já pré-processadas (pp) e recorta apenas a ROI da vaga
    para cada feature. Muito mais eficiente do que re-processar a imagem por vaga.

    Devolve dict {'f1': float, 'f2': float, 'f3': float, 'f4': float}
    ou None se a ROI for demasiado pequena para classificar com confiança.
    """
    cfg = PARK_CFG[parque]
    x, y, w, h = ann['bbox']
    x1, y1 = int(x), int(y)
    # Clipar coordenadas para não sair fora dos limites da imagem
    x2, y2 = min(int(x + w), W), min(int(y + h), H)
    if (y2 - y1) < 4 or (x2 - x1) < 4:  # ROI demasiado pequena → ignorar
        return None
    feats = {}
    for fi, (p, feat, _) in enumerate(cfg['feats'], 1):
        roi = pp[p][y1:y2, x1:x2]        # recortar ROI da imagem pré-processada
        if roi.size < 16:                 # GLCM instável com menos de 16 píxeis
            return None
        feats[f'f{fi}'] = FEAT_FUNCS[feat](roi, med_global=med_g.get(p))
    # F4 — Harris corners: usa o seu próprio pipeline e parâmetros per-parque
    roi_f4 = pp[cfg['f4_pipeline']][y1:y2, x1:x2]
    feats['f4'] = n_cantos(roi_f4, cfg['f4_thresh'], cfg['f4_min_dist'])
    return feats

def predizer(feats, calib):
    """
    Aplica a votacao maioritária com os limiares do passo3_limiares.json.

    Para cada feature activa:
      - 'gt': vota OC se valor > threshold  (ex: Sobel alto → bordas → carro)
      - 'lt': vota OC se valor < threshold  (ex: homogeneidade baixa → textura → carro)

    votos_n >= min_votos  →  pred = 1 (OCUPADA)
    votos_n <  min_votos  →  pred = 0 (LIVRE)

    Devolve (pred, n_votos_OC, detalhe_por_feature)
    O detalhe é mostrado no painel lateral quando o utilizador clica numa vaga.
    """
    votos_n  = 0
    votos_dt = {}
    fi_usar  = calib.get('feats_usar', ['f1', 'f2', 'f3', 'f4'])
    for fi in fi_usar:
        t, d = calib['thresholds'][fi], calib['directions'][fi]
        v    = feats.get(fi, 0.0)
        ok   = (v > t) if d == 'gt' else (v < t)   # este filtro vota OC?
        votos_n += int(ok)
        # Guardar detalhe para mostrar na UI (valor, limiar, voto)
        votos_dt[fi] = {'val': v, 't': t, 'd': d, 'voto': 'OC' if ok else 'LV'}
    pred = 1 if votos_n >= calib['min_votos'] else 0
    return pred, votos_n, votos_dt


# ═══════════════════════════════════════════════════════════════════════════════
#  APLICACAO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class ParqueApp:

    def __init__(self, root):
        self.root = root
        self.root.title(
            "Classificador de Vagas de Parque  |  Passo 6  |  MEEC / IPB 2025-2026")
        self.root.geometry("1300x820")
        self.root.minsize(960, 640)
        self.root.configure(bg=BG)

        # Estado
        self.img_path     = None
        self.img_pil      = None      # PIL RGB (sem overlay)
        self.img_display  = None      # PIL RGB (com overlay)
        self.photo_tk     = None      # PhotoImage atual
        self.scale        = 1.0
        self.parque       = None
        self.split_name   = None
        self.anns         = []
        self.results      = []        # [{ann, feats, pred, gt, ...}]
        self.sel_idx      = None
        self.coco_lut     = {}        # filename -> {split, anns, img_dir}
        self.calib        = {}
        self.classif_done = False
        self._classif_lock = threading.Lock()
        self.extra_vagas  = []        # posicoes canonicas extra G100 (crop)
        self.extra_results = []       # classificacao das vagas extra
        self.gray_img      = None      # imagem cinzenta (para pipeline viz)
        self._sel_extra    = False     # ultima seleccao foi vaga extra?
        self._sel_extra_idx = None

        self._load_coco_data()
        self._load_calib()
        self._load_extra_vagas()
        self._build_ui()
        self._status("Pronto — Abra uma imagem do dataset  (Ficheiro > Abrir Imagem)")

    # ── Carregar dados ao iniciar ────────────────────────────────────────────
    def _load_coco_data(self):
        # 1) Dataset original (G28, G40, G100)
        for split in ('train', 'valid', 'test'):
            path = os.path.join(DATASET, split, '_annotations.coco.json')
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            anns_d = {}
            for ann in data['annotations']:
                anns_d.setdefault(ann['image_id'], []).append(ann)
            for img in data['images']:
                fn = img['file_name']
                self.coco_lut[fn] = {
                    'split'  : split,
                    'anns'   : anns_d.get(img['id'], []),
                    'img_dir': os.path.join(DATASET, split),
                }

        # 2) Dataset unificado (crop folder) -- sobrescreve img_dir para todas as imagens presentes
        #    G100 usa coordenadas ajustadas (crop 575x371); G28/G40 usam coords originais (640x640)
        if os.path.isdir(DATASET_G100):
            for split in ('train', 'valid', 'test'):
                path = os.path.join(DATASET_G100, split, '_annotations.coco.json')
                if not os.path.exists(path):
                    continue
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                anns_d = {}
                for ann in data['annotations']:
                    anns_d.setdefault(ann['image_id'], []).append(ann)
                for img in data['images']:
                    fn   = img['file_name']
                    anns = anns_d.get(img['id'], [])
                    self.coco_lut[fn] = {
                        'split'  : split,
                        'anns'   : anns,
                        'img_dir': os.path.join(DATASET_G100, split),
                    }

    def _load_calib(self):
        with open(CALIB_FILE, encoding='utf-8') as f:
            raw = json.load(f)
        self.calib = {p: raw[p] for p in ('G28', 'G40', 'G100')}

    def _load_extra_vagas(self):
        if not os.path.exists(EXTRA_VAGAS_FILE):
            return
        with open(EXTRA_VAGAS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        self.extra_vagas = data.get('vagas', [])

    # ── Construir interface ──────────────────────────────────────────────────
    def _build_ui(self):
        self._build_toolbar()
        self._build_main()
        self._build_statusbar()

    def _build_toolbar(self):
        tb = tk.Frame(self.root, bg=BG_TB, height=52)
        tb.pack(fill='x', side='top')
        tb.pack_propagate(False)

        kw = dict(bg='#313244', fg=FG, relief='flat', padx=12, pady=6,
                  font=('Segoe UI', 10), cursor='hand2',
                  activebackground='#45475a', activeforeground=FG, bd=0)

        tk.Button(tb, text='\U0001F4C2  Abrir Imagem',
                  command=self.abrir_imagem, **kw).pack(
            side='left', padx=(12, 6), pady=10)

        self.btn_classif = tk.Button(tb, text='▶  Classificar',
                                     command=self.classificar,
                                     state='disabled', **kw)
        self.btn_classif.pack(side='left', padx=6, pady=10)

        tk.Frame(tb, bg='#45475a', width=1).pack(side='left', fill='y', pady=10, padx=10)

        tk.Button(tb, text='🗺  Como mapear vagas?',
                  command=self._abrir_mapeamento, **kw).pack(
            side='left', padx=6, pady=10)

        tk.Frame(tb, bg='#45475a', width=1).pack(side='left', fill='y', pady=10, padx=10)

        tk.Button(tb, text='➕', command=lambda: self.zoom(1.2), **kw).pack(
            side='left', padx=2, pady=10)
        tk.Button(tb, text='➖', command=lambda: self.zoom(0.833), **kw).pack(
            side='left', padx=2, pady=10)
        tk.Button(tb, text='Ajustar', command=self.zoom_fit, **kw).pack(
            side='left', padx=2, pady=10)
        tk.Button(tb, text='100%',
                  command=lambda: self._set_scale(1.0), **kw).pack(
            side='left', padx=2, pady=10)

        self.lbl_zoom = tk.Label(tb, text='100%', bg=BG_TB, fg=FG_DIM,
                                 font=('Segoe UI', 9))
        self.lbl_zoom.pack(side='left', padx=8)

        self.lbl_park_tb = tk.Label(tb, text='', bg=BG_TB, fg=ACCENT,
                                    font=('Segoe UI', 11, 'bold'))
        self.lbl_park_tb.pack(side='right', padx=16)

    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True)

        # ── Canvas (esquerda) ────────────────────────────────────────────────
        cf = tk.Frame(main, bg=BG_CANVAS)
        cf.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(cf, bg=BG_CANVAS, cursor='crosshair',
                                highlightthickness=0)
        vsb = tk.Scrollbar(cf, orient='vertical',   command=self.canvas.yview)
        hsb = tk.Scrollbar(cf, orient='horizontal', command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.canvas.pack(fill='both', expand=True)

        # Bindings
        self.canvas.bind('<Button-1>',       self.on_canvas_click)
        self.canvas.bind('<Control-MouseWheel>', self.on_ctrl_scroll)
        self.canvas.bind('<MouseWheel>',     self.on_scroll_y)
        self.canvas.bind('<Shift-MouseWheel>', self.on_scroll_x)
        self.canvas.bind('<Button-4>',       lambda e: self.canvas.yview_scroll(-1, 'units'))
        self.canvas.bind('<Button-5>',       lambda e: self.canvas.yview_scroll( 1, 'units'))

        # Placeholder centralizado
        self._canvas_placeholder()

        # ── Painel direito ───────────────────────────────────────────────────
        panel = tk.Frame(main, bg=BG_PANEL, width=385)
        panel.pack(side='right', fill='y')
        panel.pack_propagate(False)
        self._build_side_panel(panel)

    def _canvas_placeholder(self):
        self.canvas.delete('all')
        self.canvas.create_text(
            400, 300,
            text='\U0001F17F  Abra uma imagem do dataset para comecar',
            fill=FG_DIM, font=('Segoe UI', 13), anchor='center')

    def _build_side_panel(self, p):
        # Helper labels
        def title(text):
            tk.Label(p, text=text, bg=BG_PANEL, fg=ACCENT,
                     font=('Segoe UI', 9, 'bold'), anchor='w').pack(
                fill='x', padx=14, pady=(12, 2))

        def sep():
            tk.Frame(p, bg=SEP, height=1).pack(fill='x', padx=14, pady=6)

        def lbl(text='', fg=FG, font=('Segoe UI', 10)):
            l = tk.Label(p, text=text, bg=BG_PANEL, fg=fg,
                         font=font, anchor='w', wraplength=345)
            l.pack(fill='x', padx=14, pady=1)
            return l

        # ── Info imagem ──────────────────────────────────────────────────────
        title('IMAGEM')
        self.lbl_split  = lbl('Split:   —', fg=FG_DIM)
        self.lbl_img    = lbl('Ficheiro: —', fg=FG_DIM)
        self.lbl_nvagas = lbl('Vagas:   —', fg=FG_DIM)
        sep()

        # ── Resultado ────────────────────────────────────────────────────────
        title('RESULTADO DA CLASSIFICACAO')
        self.lbl_livre = lbl('Livre:     —', fg=OK_G,  font=('Segoe UI', 13, 'bold'))
        self.lbl_ocup  = lbl('Ocupada:   —', fg=ERR_R, font=('Segoe UI', 13, 'bold'))
        sep()

        # ── Avaliacao ────────────────────────────────────────────────────────
        title('AVALIACAO  (vs. ground truth)')
        self.lbl_acc   = lbl('Accuracy:   —', font=('Segoe UI', 13, 'bold'))
        self.lbl_tp_fp = lbl('TP: —   FP: —', fg=FG_DIM)
        self.lbl_fn_tn = lbl('FN: —   TN: —', fg=FG_DIM)
        sep()

        # ── Legenda ──────────────────────────────────────────────────────────
        title('LEGENDA')
        for cor, txt in [
            (OK_G,  '■ Verde  = Livre'),
            (ERR_R, '■ Verm.  = Ocupada'),
            (FG,    '□ Branco = Vaga selecionada'),
        ]:
            tk.Label(p, text=f'  {txt}', bg=BG_PANEL, fg=cor,
                     font=('Segoe UI', 9), anchor='w').pack(
                fill='x', padx=14, pady=0)
        sep()

        # ── Detalhe spot ─────────────────────────────────────────────────────
        title('VAGA SELECIONADA  (clique para ver)')
        self.txt_det = tk.Text(
            p, bg='#11111b', fg=FG, font=('Consolas', 9),
            height=11, relief='flat', wrap='word', state='disabled',
            padx=8, pady=6, insertbackground=FG)
        self.txt_det.pack(fill='x', padx=14, pady=(4, 6))
        self.txt_det.tag_configure('oc',    foreground=ERR_R)
        self.txt_det.tag_configure('lv',    foreground=OK_G)
        self.txt_det.tag_configure('bold',  font=('Consolas', 9, 'bold'))

        # Botao "Como foi avaliado"
        self.btn_pipeline = tk.Button(
            p, text='🔍  Como foi avaliado?',
            bg='#313244', fg=ACCENT,
            font=('Segoe UI', 10, 'bold'),
            relief='flat', cursor='hand2',
            activebackground='#45475a', activeforeground=ACCENT,
            state='disabled',
            command=self._abrir_pipeline)
        self.btn_pipeline.pack(fill='x', padx=14, pady=(0, 10))
        self.txt_det.tag_configure('dim',   foreground=FG_DIM)
        self._det_placeholder()

    def _build_statusbar(self):
        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var, bg='#12121c',
                 fg=FG_DIM, font=('Segoe UI', 9), anchor='w', padx=10
                 ).pack(fill='x', side='bottom')

    # ── Abrir imagem ─────────────────────────────────────────────────────────
    def abrir_imagem(self):
        # Preferir pasta G100 crop se existir, senão dataset original
        init_dir = os.path.join(DATASET_G100, 'test')
        if not os.path.isdir(init_dir):
            init_dir = os.path.join(DATASET, 'test')
        if not os.path.isdir(init_dir):
            init_dir = DATASET
        path = filedialog.askopenfilename(
            title='Abrir imagem do parking lot',
            initialdir=init_dir,
            filetypes=[('Imagens', '*.jpg *.jpeg *.png'), ('Todos', '*.*')],
        )
        if not path:
            return

        fn = os.path.basename(path)
        info = self.coco_lut.get(fn)
        if info is None:
            messagebox.showwarning(
                'Imagem nao encontrada',
                f'"{fn}" nao esta nas anotacoes COCO.\n\n'
                'Escolha uma imagem das pastas:\n'
                '  dataset_parkinglot/train/\n'
                '  dataset_parkinglot/valid/\n'
                '  dataset_parkinglot/test/')
            return

        anns = info['anns']
        n_v  = len(anns)
        parque = N_VAGAS_TO_PARQUE.get(n_v)
        if parque is None:
            messagebox.showwarning(
                'Parque nao reconhecido',
                f'{n_v} vagas anotadas -- nao corresponde a G28 (28), G40 (40) ou G100 (100).')
            return

        # Caminho real da imagem (pode ser o crop para G100)
        actual_path = os.path.join(info['img_dir'], fn)
        if not os.path.exists(actual_path):
            actual_path = path   # fallback para o path seleccionado

        # Guardar estado
        self.img_path     = actual_path
        self.parque       = parque
        self.split_name    = info['split']
        self.anns          = anns
        self.results       = []
        self.extra_results = []
        self.classif_done  = False
        self.sel_idx       = None
        self._sel_extra    = False
        self._sel_extra_idx = None
        self.gray_img      = None
        if hasattr(self, 'btn_pipeline'):
            self.btn_pipeline.configure(state='disabled')

        # Carregar PIL sem overlay
        raw = imread(actual_path)
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        elif raw.shape[2] == 4:
            raw = raw[:, :, :3]
        if raw.dtype != np.uint8:
            raw = (raw * 255).clip(0, 255).astype(np.uint8)
        self.img_pil     = Image.fromarray(raw)
        self.img_display = self.img_pil.copy()

        # Atualizar painel
        cor_p = PARK_CFG[parque]['cor']
        self.lbl_park_tb.configure(text=f'  Parque: {parque}  ',
                                   fg=cor_p)
        self.lbl_split.configure( text=f'Split:    {info["split"]}')
        self.lbl_img.configure(   text=f'Ficheiro: {fn[:42]}{"..." if len(fn)>42 else ""}')
        self.lbl_nvagas.configure(text=f'Vagas:    {n_v}  ({parque})')
        self.lbl_livre.configure( text='Livre:     —  (clique Classificar)')
        self.lbl_ocup.configure(  text='Ocupada:   —  (clique Classificar)')
        self.lbl_acc.configure(   text='Accuracy:   —', fg=FG)
        self.lbl_tp_fp.configure( text='TP: —   FP: —')
        self.lbl_fn_tn.configure( text='FN: —   TN: —')
        self._det_placeholder()
        self.btn_classif.configure(state='normal')

        # Mostrar imagem + zoom ajustado
        self.root.after(50, self._zoom_fit_delayed)
        self._status(f'Imagem carregada: {fn}  |  {parque}  |  {n_v} vagas  |  {info["split"]}')

    def _zoom_fit_delayed(self):
        self._render_canvas()
        self.zoom_fit()

    # ── Classificar ──────────────────────────────────────────────────────────
    def classificar(self):
        if self.img_path is None:
            return
        self.btn_classif.configure(state='disabled', text='⏳  A classificar...')
        self._status(f'A classificar {len(self.anns)} vagas ({self.parque})... aguarde.')
        threading.Thread(target=self._thread_classif, daemon=True).start()

    def _thread_classif(self):
        t0 = time.time()
        try:
            gray = rgb2gray(imread(self.img_path))
            H, W = gray.shape
            pp, med_g = processar_img(gray, self.parque)
            calib     = self.calib[self.parque]

            # ── Vagas COCO (com ground truth) ──────────────────────────────
            results = []
            for ann in self.anns:
                feats = extrair_feats(pp, med_g, self.parque, ann, H, W)
                if feats is None:
                    continue
                pred, n_v, vd = predizer(feats, calib)
                gt = 0 if ann['category_id'] == CAT_LIVRE else 1
                results.append({
                    'ann'     : ann,
                    'feats'   : feats,
                    'pred'    : pred,
                    'gt'      : gt,
                    'correto' : int(pred == gt),
                    'n_votos' : n_v,
                    'votos_dt': vd,
                    'extra'   : False,
                })

            # ── Vagas extra G100 (sem ground truth) ──────────────────────────
            extra_results = []
            if self.parque == 'G100' and self.extra_vagas:
                H2, W2 = pp[list(pp.keys())[0]].shape
                for v in self.extra_vagas:
                    ann_like = {'bbox': [v['x'], v['y'], v['w'], v['h']],
                                'category_id': -1}
                    feats = extrair_feats(pp, med_g, 'G100', ann_like, H2, W2)
                    if feats is None:
                        continue
                    pred, n_v, vd = predizer(feats, calib)
                    extra_results.append({
                        'ann'     : ann_like,
                        'feats'   : feats,
                        'pred'    : pred,
                        'gt'      : -1,
                        'correto' : -1,
                        'n_votos' : n_v,
                        'votos_dt': vd,
                    })

            elapsed = time.time() - t0
            with self._classif_lock:
                self.results       = results
                self.extra_results = extra_results
                self.gray_img      = gray      # guardar para pipeline viz
                self.classif_done  = True
            self.root.after(0, lambda: self._pos_classif(elapsed))
        except Exception as exc:
            self.root.after(0, lambda: self._classif_erro(str(exc)))

    def _pos_classif(self, elapsed):
        self._draw_overlay()
        self._update_stats()
        self.btn_classif.configure(state='normal', text='▶  Classificar')
        n   = len(self.results)
        acc = sum(r['correto'] for r in self.results) / max(1, n) * 100
        self._status(
            f'Classificacao concluida  |  {n} vagas  |  '
            f'Accuracy: {acc:.1f}%  |  {elapsed:.1f}s')

    def _classif_erro(self, msg):
        self.btn_classif.configure(state='normal', text='▶  Classificar')
        messagebox.showerror('Erro na classificacao', msg)

    # ── Overlay PIL ───────────────────────────────────────────────────────────
    def _draw_overlay(self, highlight_idx=None, highlight_extra_idx=None):
        if self.img_pil is None:
            return
        base    = self.img_pil.convert('RGBA')
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        # Vagas COCO — borda solida
        for i, r in enumerate(self.results):
            x, y, w, h = r['ann']['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            fill = COL_LIVRE if r['pred'] == 0 else COL_OCUP
            brd  = BRD_LIVRE if r['pred'] == 0 else BRD_OCUP
            sel  = (i == highlight_idx)
            bw, bc = (4, (255,255,255)) if sel else (2, brd)
            draw.rectangle([x1,y1,x2,y2], fill=fill)
            for b in range(bw):
                draw.rectangle([x1-b,y1-b,x2+b,y2+b], outline=bc)

        # Vagas extra — borda solida igual às COCO
        for i, r in enumerate(self.extra_results):
            x, y, w, h = r['ann']['bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            fill = COL_LIVRE if r['pred'] == 0 else COL_OCUP
            brd  = BRD_LIVRE if r['pred'] == 0 else BRD_OCUP
            sel  = (i == highlight_extra_idx)
            bw, bc = (4, (255,255,255)) if sel else (2, brd)
            draw.rectangle([x1,y1,x2,y2], fill=fill)
            for b in range(bw):
                draw.rectangle([x1-b,y1-b,x2+b,y2+b], outline=bc)

        self.img_display = Image.alpha_composite(base, overlay).convert('RGB')
        self._render_canvas()

    def _render_canvas(self):
        if self.img_display is None:
            return
        W0, H0 = self.img_display.size
        Wd = max(1, int(W0 * self.scale))
        Hd = max(1, int(H0 * self.scale))
        scaled = self.img_display.resize((Wd, Hd), Image.LANCZOS)
        self.photo_tk = ImageTk.PhotoImage(scaled)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo_tk)
        self.canvas.configure(scrollregion=(0, 0, Wd, Hd))

    # ── Zoom ─────────────────────────────────────────────────────────────────
    def zoom(self, factor):
        self._set_scale(self.scale * factor)

    def zoom_fit(self):
        if self.img_display is None:
            return
        cw = max(self.canvas.winfo_width(),  100)
        ch = max(self.canvas.winfo_height(), 100)
        W0, H0 = self.img_display.size
        self._set_scale(min(cw / W0, ch / H0, 1.0))

    def _set_scale(self, s):
        self.scale = max(0.05, min(10.0, s))
        self.lbl_zoom.configure(text=f'{self.scale * 100:.0f}%')
        self._render_canvas()

    def on_ctrl_scroll(self, e):
        self.zoom(1.1 if e.delta > 0 else 0.909)

    def on_scroll_y(self, e):
        if e.state & 0x0004:   # Ctrl pressionado
            self.zoom(1.1 if e.delta > 0 else 0.909)
        else:
            self.canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units')

    def on_scroll_x(self, e):
        self.canvas.xview_scroll(-1 if e.delta > 0 else 1, 'units')

    # ── Click numa vaga ───────────────────────────────────────────────────────
    def on_canvas_click(self, event):
        if not self.classif_done:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix = cx / self.scale
        iy = cy / self.scale

        # Verificar COCO primeiro, depois extra
        clicked = clicked_extra = None
        for i, r in enumerate(self.results):
            x, y, w, h = r['ann']['bbox']
            if x <= ix <= x+w and y <= iy <= y+h:
                clicked = i; break
        if clicked is None:
            for i, r in enumerate(self.extra_results):
                x, y, w, h = r['ann']['bbox']
                if x <= ix <= x+w and y <= iy <= y+h:
                    clicked_extra = i; break

        if clicked is not None:
            self.sel_idx       = clicked
            self._sel_extra    = False
            self._sel_extra_idx = None
            self._mostrar_detalhe(clicked, extra=False)
            self._draw_overlay(highlight_idx=clicked)
            if hasattr(self, 'btn_pipeline'):
                self.btn_pipeline.configure(state='normal')
        elif clicked_extra is not None:
            self.sel_idx        = None
            self._sel_extra     = True
            self._sel_extra_idx = clicked_extra
            self._mostrar_detalhe(clicked_extra, extra=True)
            self._draw_overlay(highlight_extra_idx=clicked_extra)
            if hasattr(self, 'btn_pipeline'):
                self.btn_pipeline.configure(state='normal')

    def _abrir_mapeamento(self):
        """
        Abre o visualizador de Canny adaptativo por faixas para demonstrar
        como as vagas de um parque podem ser detectadas automaticamente
        numa imagem vazia, sem anotacoes COCO.
        Usa a imagem actualmente carregada (ou a imagem de exemplo se nenhuma estiver aberta).
        """
        if not _CANNY_OK:
            messagebox.showwarning(
                'Script indisponivel',
                f'Nao foi encontrado passo_teste_canny_interativo.py em:\n{_CANNY_SCRIPT}'
            )
            return
        img_arg = self.img_path if self.img_path else ''
        subprocess.Popen([sys.executable, _CANNY_SCRIPT, img_arg])

    def _abrir_pipeline(self):
        """Abre a janela de pipeline para a vaga actualmente seleccionada."""
        if not _PIPELINE_OK:
            messagebox.showwarning('Pipeline indisponivel',
                f'Nao foi encontrado passo7_pipeline_viz.py em:\n{_PIPELINE_SCRIPT}')
            return
        if self.gray_img is None:
            messagebox.showinfo('Sem imagem', 'Classifique uma imagem primeiro.')
            return

        # Determinar vaga seleccionada
        if self._sel_extra and self._sel_extra_idx is not None:
            r = self.extra_results[self._sel_extra_idx]
            gt_str = 'extra'
        elif self.sel_idx is not None:
            r = self.results[self.sel_idx]
            gt_str = 'LV' if r['gt'] == 0 else 'OC'
        else:
            messagebox.showinfo('Nenhuma vaga', 'Clique numa vaga primeiro.')
            return

        x, y, w, h = [int(v) for v in r['ann']['bbox']]

        # Guardar imagem cinzenta num ficheiro temporario
        tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(tmp.name, self.gray_img)
        tmp.close()

        # Lançar subprocess independente (tem o seu proprio main loop tkinter)
        subprocess.Popen([
            sys.executable, _PIPELINE_SCRIPT,
            tmp.name, str(x), str(y), str(w), str(h), gt_str
        ])

    def _mostrar_detalhe(self, idx, extra=False):
        r   = (self.extra_results if extra else self.results)[idx]
        ann = r['ann']
        bx, by, bw, bh = ann['bbox']
        pred_str = 'LIVRE' if r['pred'] == 0 else 'OCUPADA'

        self.txt_det.configure(state='normal')
        self.txt_det.delete('1.0', 'end')

        def ins(txt, tag=None):
            self.txt_det.insert('end', txt, tag or '')

        prefixo = 'Extra #' if extra else 'Vaga #'
        ins(f'{prefixo}{idx+1}  ', 'bold')
        ins(f'bbox ({int(bx)},{int(by)}) {int(bw)}x{int(bh)} px\n', 'dim')
        ins(f'Pred: ', 'bold'); ins(f'{pred_str}\n', 'oc' if r['pred'] else 'lv')
        if not extra:
            gt_str = 'livre' if r['gt'] == 0 else 'ocupada'
            ok_str = 'CORRETO' if r['correto'] else 'ERRO'
            ins(f'GT: {gt_str}   '); ins(f'[{ok_str}]\n', 'bold')
        ins(f'Votos OC: {r["n_votos"]}/4\n')
        ins('─' * 32 + '\n', 'dim')

        cfg = PARK_CFG[self.parque]
        names = [d for _, _, d in cfg['feats']] + ['n_cantos [' + cfg['f4_pipeline'] + ']']
        for i, (fi, info) in enumerate(r['votos_dt'].items()):
            nm  = names[i] if i < len(names) else fi
            op  = '>' if info['d'] == 'gt' else '<'
            tag = 'oc' if info['voto'] == 'OC' else 'lv'
            ins(f'{fi}: ', 'bold')
            ins(f'{info["val"]:.5f}', tag)
            ins(f'  {op} {info["t"]:.5f}  ')
            ins(f'{info["voto"]}\n', tag)
            ins(f'     {nm}\n', 'dim')

        self.txt_det.configure(state='disabled')

    def _det_placeholder(self):
        self.txt_det.configure(state='normal')
        self.txt_det.delete('1.0', 'end')
        self.txt_det.insert(
            'end',
            'Clique numa vaga colorida para ver\nos valores dos 4 features e os\nvotos individuais do classificador.',
            'dim')
        self.txt_det.configure(state='disabled')

    # ── Estatisticas ─────────────────────────────────────────────────────────
    def _update_stats(self):
        res  = self.results
        n    = len(res)
        n_lv = sum(1 for r in res if r['pred'] == 0)
        n_oc = sum(1 for r in res if r['pred'] == 1)
        self.lbl_livre.configure(
            text=f'Livre:     {n_lv:3d}  ({100*n_lv/max(1,n):.1f}%)')
        self.lbl_ocup.configure(
            text=f'Ocupada:   {n_oc:3d}  ({100*n_oc/max(1,n):.1f}%)')

        tp  = sum(1 for r in res if r['pred'] == 1 and r['gt'] == 1)
        tn  = sum(1 for r in res if r['pred'] == 0 and r['gt'] == 0)
        fp  = sum(1 for r in res if r['pred'] == 1 and r['gt'] == 0)
        fn  = sum(1 for r in res if r['pred'] == 0 and r['gt'] == 1)
        acc = (tp + tn) / max(1, n) * 100
        cor = OK_G if acc >= 80 else (WARN_O if acc >= 70 else ERR_R)
        self.lbl_acc.configure(  text=f'Accuracy:   {acc:.1f}%', fg=cor)
        self.lbl_tp_fp.configure(text=f'TP: {tp:3d}   FP: {fp:3d}')
        self.lbl_fn_tn.configure(text=f'FN: {fn:3d}   TN: {tn:3d}')


    # ── Util ─────────────────────────────────────────────────────────────────
    def _status(self, msg):
        self.status_var.set(f'  {msg}')


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.25)   # DPI scaling suave no Windows
    except Exception:
        pass
    app = ParqueApp(root)
    root.mainloop()
