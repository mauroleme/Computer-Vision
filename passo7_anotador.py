# passo7_anotador.py
# Ferramenta semi-automatica de anotacao de vagas G100
# Passo 7 -- Visao por Computador -- MEEC / IPB 2025-2026
# Aluno: Mauro da Silva Leme (a52965)
#
# COMO USAR:
#   1. Abrir uma imagem G100 (train/valid/test)
#   2. Modo "Linha": clicar + arrastar horizontalmente por cima de uma fila de vagas
#   3. O sistema gera bboxes e classifica cada vaga automaticamente
#   4. Clicar numa vaga para confirmar ou inverter o label
#   5. "Exportar CSV" guarda os novos labels para uso futuro
#
# TECLAS:
#   L         = activar modo Linha
#   C         = activar modo Confirmar
#   Ctrl+Z    = desfazer ultima linha
#   Ctrl+S    = guardar CSV
#   +/-       = zoom
#   F         = ajustar zoom

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import json, os, csv, warnings, time
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, prewitt, gaussian, unsharp_mask
from skimage.filters import median as sk_median
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops, corner_harris, corner_peaks

warnings.filterwarnings('ignore')

# ── Caminhos ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET      = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot'))
DATASET_G100 = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'dataset_parkinglot_g100_crop'))
CALIB_FILE   = os.path.join(_SCRIPT_DIR, 'passo3_limiares.json')
EXPORT_DIR   = os.path.join(_SCRIPT_DIR, 'vagas_indefinidas')
os.makedirs(EXPORT_DIR, exist_ok=True)

CAT_LIVRE   = 1
CAT_OCUPADA = 2

# ── Tema ──────────────────────────────────────────────────────────────────────
BG        = '#1e1e2e'
BG_PANEL  = '#181825'
BG_CANVAS = '#11111b'
BG_TB     = '#12121c'
FG        = '#cdd6f4'
FG_DIM    = '#6c7086'
ACCENT    = '#89b4fa'
SEP       = '#313244'

# Cores overlay (PIL RGBA)
RGBA_LV_AUTO  = (  0, 210,  60, 90)
RGBA_OC_AUTO  = (220,  40,  40, 90)
RGBA_LV_CONF  = (  0, 255,  80, 150)
RGBA_OC_CONF  = (255,  50,  50, 150)
RGBA_EXIST_LV = ( 60, 180,  60, 60)
RGBA_EXIST_OC = (180,  60,  60, 60)
RGBA_GUIDE    = (255, 200,   0, 180)

BRD_LV_AUTO  = (  0, 210,  60)
BRD_OC_AUTO  = (220,  40,  40)
BRD_LV_CONF  = (  0, 255,  80)
BRD_OC_CONF  = (255,  50,  50)
BRD_GUIDE    = (255, 200,   0)
BRD_EXIST    = (100, 100, 100)
BRD_SELECT   = (255, 255, 255)

# ── Classificador G100 (identico ao passo4) ───────────────────────────────────
T1, T2, T3, T4 = 0.567394, 0.042270, 0.040427, 0.0
MIN_VOTOS = 3
_disk1 = disk(1)

def _unsharp(g):
    return np.clip(unsharp_mask(g, radius=2, amount=1.0), 0.0, 1.0)

PIPELINE_FUNCS = {
    'raw'          : lambda g: g,
    'gauss'        : lambda g: gaussian(g, sigma=1),
    'median'       : lambda g: sk_median(g, footprint=_disk1).astype(float),
    'unsharp'      : _unsharp,
    'median_gauss' : lambda g: gaussian(sk_median(g, footprint=_disk1), sigma=1.0),
    'gauss_median' : lambda g: sk_median(gaussian(g, sigma=1.0), footprint=_disk1).astype(float),
    'unsharp_gauss': lambda g: gaussian(_unsharp(g), sigma=1),
    'gauss_unsharp': lambda g: np.clip(unsharp_mask(gaussian(g, sigma=1), radius=2, amount=1.0), 0.0, 1.0),
}

def processar(gray):
    mg = PIPELINE_FUNCS['median_gauss'](gray)
    gm = PIPELINE_FUNCS['gauss_median'](gray)
    me = PIPELINE_FUNCS['median'](gray)
    return mg, gm, me

def glcm_hom(roi):
    q = (np.clip(roi, 0, 1) * 63).astype(np.uint8)
    g = graycomatrix(q, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels=64, symmetric=True, normed=True)
    return float(graycoprops(g, 'homogeneity')[0].mean())

def classificar_roi(mg, gm, me, x, y, w, h):
    H, W = mg.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(x + w, W), min(y + h, H)
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return 'OC', 4
    roi_mg = mg[y1:y2, x1:x2]
    roi_gm = gm[y1:y2, x1:x2]
    roi_me = me[y1:y2, x1:x2]
    if roi_mg.size < 9:
        return 'OC', 4
    f1 = glcm_hom(roi_mg)
    f2 = float(np.mean(np.abs(sobel(roi_gm))))
    f3 = float(np.mean(np.abs(prewitt(roi_mg))))
    try:
        f4 = float(len(corner_peaks(corner_harris(roi_me),
                                    min_distance=3, threshold_rel=0.05)))
    except Exception:
        f4 = 0.0
    n_oc = int(f1 < T1) + int(f2 > T2) + int(f3 > T3) + int(f4 > T4)
    return ('OC' if n_oc >= MIN_VOTOS else 'LV'), n_oc


# ── Modelo de perspectiva padrao G100 (calibrado em passo7_row_analysis) ─────
# dx(y) = 0.0343*y + 10.98   w(y) = 0.0468*y + 10.51   h(y) = 0.0588*y + 29.13
_PERSP = {'a_dx': 0.0343, 'b_dx': 10.98,
           'a_w':  0.0468, 'b_w':  10.51,
           'a_h':  0.0588, 'b_h':  29.13}

def perspectiva(y, persp=None):
    p = persp or _PERSP
    dx = max(8.0, p['a_dx'] * y + p['b_dx'])
    w  = max(6.0, p['a_w']  * y + p['b_w'])
    h  = max(6.0, p['a_h']  * y + p['b_h'])
    return dx, w, h


# ═══════════════════════════════════════════════════════════════════════════════
#  APLICACAO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════
class AnotadorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Anotador de Vagas G100  |  Passo 7  |  MEEC / IPB 2025-2026")
        self.root.geometry("1300x820")
        self.root.minsize(960, 640)
        self.root.configure(bg=BG)

        # ── Estado ────────────────────────────────────────────────────────────
        self.img_path    = None
        self.img_pil     = None          # PIL RGB original
        self.img_display = None          # PIL RGB com overlays
        self.photo_tk    = None
        self.scale       = 1.0
        self.img_w = self.img_h = 640

        self.gray_img = None             # numpy grayscale
        self.pp_mg = self.pp_gm = self.pp_me = None  # pipeline images
        self.persp   = None              # modelo de perspectiva da imagem atual

        self.exist_spots = []            # vagas ja anotadas no COCO [{x,y,w,h,gt}]
        self.novo_spots  = []            # novas vagas [{x,y,w,h,pred,label,confirmed}]
        self.guias       = []            # linhas guia desenhadas [{x1,y1,x2,y2}]
        self.sel_idx     = None          # indice de nova vaga selecionada
        self.coco_lut    = {}

        # ── Modo ──────────────────────────────────────────────────────────────
        self.mode      = tk.StringVar(value='linha')  # 'linha' | 'ponto' | 'confirmar'
        self.scale_var = tk.DoubleVar(value=1.0)      # factor de escala manual

        # ── Drawing state (modo Linha) ─────────────────────────────────────────
        self._drawing    = False
        self._draw_start = None          # (img_x, img_y)
        self._guide_id   = None          # canvas item id da guia temporaria

        # ── Ponto state (modo Ponto) ──────────────────────────────────────────
        self.placing     = None          # {cx, cy, w, h, rot} — vaga a colocar
        self.rot_deg     = 0.0           # rotacao actual (graus)

        self._load_coco()
        self._build_ui()
        self._status("Pronto — Abra uma imagem G100 e comece a desenhar linhas")

    # ── Carregar COCO lookup ──────────────────────────────────────────────────
    def _load_coco(self):
        # 1) Dataset original (todos os parques)
        for split in ('train', 'valid', 'test'):
            path = os.path.join(DATASET, split, '_annotations.coco.json')
            if not os.path.exists(path):
                continue
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            ad = {}
            for ann in data['annotations']:
                ad.setdefault(ann['image_id'], []).append(ann)
            for img in data['images']:
                self.coco_lut[img['file_name']] = {
                    'split': split, 'anns': ad.get(img['id'], []),
                    'img_dir': os.path.join(DATASET, split),
                }

        # 2) Dataset G100 cortado — sobrescreve entradas G100 com coords ajustadas
        if os.path.isdir(DATASET_G100):
            for split in ('train', 'valid', 'test'):
                path = os.path.join(DATASET_G100, split, '_annotations.coco.json')
                if not os.path.exists(path):
                    continue
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                ad = {}
                for ann in data['annotations']:
                    ad.setdefault(ann['image_id'], []).append(ann)
                for img in data['images']:
                    anns = ad.get(img['id'], [])
                    if len(anns) == 100:
                        self.coco_lut[img['file_name']] = {
                            'split': split, 'anns': anns,
                            'img_dir': os.path.join(DATASET_G100, split),
                        }

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_toolbar()
        self._build_main()
        self._build_statusbar()
        self.root.bind('<l>', lambda e: self.mode.set('linha'))
        self.root.bind('<p>', lambda e: self.mode.set('ponto'))
        self.root.bind('<c>', lambda e: self.mode.set('confirmar'))
        self.root.bind('<Control-z>', lambda e: self.desfazer())
        self.root.bind('<Control-s>', lambda e: self.exportar())
        self.root.bind('<plus>',  lambda e: self.zoom(1.2))
        self.root.bind('<minus>', lambda e: self.zoom(0.833))
        self.root.bind('<f>',     lambda e: self.zoom_fit())
        # Modo Ponto: setas = mover 1px, Shift+setas = mover 5px
        self.root.bind('<Left>',         lambda e: self._nudge_ponto(-1,  0))
        self.root.bind('<Right>',        lambda e: self._nudge_ponto(+1,  0))
        self.root.bind('<Up>',           lambda e: self._nudge_ponto( 0, -1))
        self.root.bind('<Down>',         lambda e: self._nudge_ponto( 0, +1))
        self.root.bind('<Shift-Left>',   lambda e: self._nudge_ponto(-5,  0))
        self.root.bind('<Shift-Right>',  lambda e: self._nudge_ponto(+5,  0))
        self.root.bind('<Shift-Up>',     lambda e: self._nudge_ponto( 0, -5))
        self.root.bind('<Shift-Down>',   lambda e: self._nudge_ponto( 0, +5))
        # , e . = rodar -1 e +1 grau
        self.root.bind('<comma>',  lambda e: self._ajustar_rot_ponto(-1))
        self.root.bind('<period>', lambda e: self._ajustar_rot_ponto(+1))
        # Enter ou Espaco = confirmar vaga
        self.root.bind('<Return>', lambda e: self._confirmar_ponto())
        self.root.bind('<space>',  lambda e: self._confirmar_ponto())
        # Escape = cancelar vaga actual
        self.root.bind('<Escape>', lambda e: self._cancelar_ponto())

    def _build_toolbar(self):
        tb = tk.Frame(self.root, bg=BG_TB, height=52)
        tb.pack(fill='x', side='top')
        tb.pack_propagate(False)

        btn_kw = dict(bg='#313244', fg=FG, relief='flat', padx=10, pady=6,
                      font=('Segoe UI', 10), cursor='hand2',
                      activebackground='#45475a', activeforeground=FG, bd=0)

        tk.Button(tb, text='\U0001F4C2  Abrir Imagem',
                  command=self.abrir, **btn_kw).pack(side='left', padx=(12,4), pady=10)

        tk.Frame(tb, bg='#45475a', width=1).pack(side='left', fill='y', pady=10, padx=8)

        # Radio buttons de modo
        for val, txt, tip in [('linha',    '📏  Linha  [L]',    'Desenhar linha para gerar vagas'),
                               ('ponto',   '🎯  Ponto  [P]',    'Clicar no centro de uma vaga para a colocar'),
                               ('confirmar','🖱  Confirmar  [C]','Clicar numa vaga para confirmar/inverter')]:
            rb = tk.Radiobutton(tb, text=txt, variable=self.mode, value=val,
                                bg=BG_TB, fg=FG, selectcolor='#45475a',
                                activebackground=BG_TB, activeforeground=FG,
                                font=('Segoe UI', 10), cursor='hand2', relief='flat',
                                indicatoron=False, padx=10, pady=6)
            rb.pack(side='left', padx=3, pady=10)

        tk.Frame(tb, bg='#45475a', width=1).pack(side='left', fill='y', pady=10, padx=8)

        tk.Button(tb, text='↩ Desfazer  [Ctrl+Z]',
                  command=self.desfazer, **btn_kw).pack(side='left', padx=3, pady=10)

        tk.Button(tb, text='💾 Exportar  [Ctrl+S]',
                  command=self.exportar, **btn_kw).pack(side='left', padx=3, pady=10)

        tk.Frame(tb, bg='#45475a', width=1).pack(side='left', fill='y', pady=10, padx=8)
        tk.Button(tb, text='➕', command=lambda: self.zoom(1.2), **btn_kw).pack(side='left', padx=2, pady=10)
        tk.Button(tb, text='➖', command=lambda: self.zoom(0.833), **btn_kw).pack(side='left', padx=2, pady=10)
        tk.Button(tb, text='Ajustar [F]', command=self.zoom_fit, **btn_kw).pack(side='left', padx=2, pady=10)

        self.lbl_zoom = tk.Label(tb, text='100%', bg=BG_TB, fg=FG_DIM, font=('Segoe UI', 9))
        self.lbl_zoom.pack(side='left', padx=8)

    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True)

        # Canvas (esquerda)
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
        self.canvas.bind('<ButtonPress-1>',   self._on_press)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<MouseWheel>',      self._on_scroll)
        self.canvas.bind('<Control-MouseWheel>', lambda e: self.zoom(1.1 if e.delta>0 else 0.9))

        # Placeholder
        self.canvas.create_text(400, 300,
            text='\U0001F17F  Abra uma imagem G100 para comecar a anotar',
            fill=FG_DIM, font=('Segoe UI', 13), anchor='center')

        # Painel direito
        panel = tk.Frame(main, bg=BG_PANEL, width=360)
        panel.pack(side='right', fill='y')
        panel.pack_propagate(False)
        self._build_panel(panel)

    def _build_panel(self, p):
        def title(t):
            tk.Label(p, text=t, bg=BG_PANEL, fg=ACCENT,
                     font=('Segoe UI', 9, 'bold'), anchor='w').pack(
                fill='x', padx=14, pady=(12, 2))
        def sep():
            tk.Frame(p, bg=SEP, height=1).pack(fill='x', padx=14, pady=6)
        def lbl(text='', fg=FG, font=('Segoe UI', 10)):
            l = tk.Label(p, text=text, bg=BG_PANEL, fg=fg,
                         font=font, anchor='w', wraplength=330)
            l.pack(fill='x', padx=14, pady=1)
            return l

        title('INSTRUCOES')
        lbl('Modo Linha: clique+arraste ao\nlongo de uma fila de vagas', fg=FG_DIM, font=('Segoe UI', 9))
        lbl('Modo Ponto [P]:\n  · Clique = colocar no centro\n  · ←↑↓→ = mover 1px\n  · Shift+seta = mover 5px\n  · , / . = rodar ±1°\n  · Enter/Espaço = confirmar\n  · Esc = cancelar', fg=FG_DIM, font=('Segoe UI', 9))
        lbl('Modo Confirmar: clique numa\nvaga para confirmar/inverter', fg=FG_DIM, font=('Segoe UI', 9))
        sep()

        # ── Rotacao (modo Ponto) ─────────────────────────────────────────────
        title('ROTACAO  [, e .]')
        rot_frame = tk.Frame(p, bg=BG_PANEL)
        rot_frame.pack(fill='x', padx=14, pady=(2, 4))

        tk.Button(rot_frame, text='−1°', bg='#313244', fg=FG, relief='flat',
                  font=('Segoe UI', 10), width=4, cursor='hand2',
                  command=lambda: self._ajustar_rot_ponto(-1)
                  ).pack(side='left', padx=(0, 4))

        self.lbl_rot = tk.Label(rot_frame, text='0.0°', bg=BG_PANEL,
                                fg=ACCENT, font=('Segoe UI', 11, 'bold'), width=6)
        self.lbl_rot.pack(side='left')

        tk.Button(rot_frame, text='+1°', bg='#313244', fg=FG, relief='flat',
                  font=('Segoe UI', 10), width=4, cursor='hand2',
                  command=lambda: self._ajustar_rot_ponto(+1)
                  ).pack(side='left', padx=(4, 8))

        tk.Button(rot_frame, text='0°', bg='#313244', fg=FG_DIM,
                  relief='flat', font=('Segoe UI', 9), cursor='hand2',
                  command=lambda: self._ajustar_rot_ponto(0, reset=True)
                  ).pack(side='left')

        self.lbl_ponto_info = lbl('—', fg=FG_DIM, font=('Consolas', 8))
        sep()

        # ── Controlo de escala ───────────────────────────────────────────────
        title('ESCALA DAS VAGAS')
        lbl('Ajustar se as vagas forem\ndemasiado grandes ou pequenas:',
            fg=FG_DIM, font=('Segoe UI', 9))

        scale_frame = tk.Frame(p, bg=BG_PANEL)
        scale_frame.pack(fill='x', padx=14, pady=(2, 4))

        tk.Button(scale_frame, text='−', bg='#313244', fg=FG, relief='flat',
                  font=('Segoe UI', 11, 'bold'), width=2, cursor='hand2',
                  command=lambda: self._ajustar_escala(-0.1)
                  ).pack(side='left', padx=(0, 4))

        self.lbl_escala = tk.Label(scale_frame, text='1.0×', bg=BG_PANEL,
                                   fg=ACCENT, font=('Segoe UI', 11, 'bold'), width=5)
        self.lbl_escala.pack(side='left')

        tk.Button(scale_frame, text='+', bg='#313244', fg=FG, relief='flat',
                  font=('Segoe UI', 11, 'bold'), width=2, cursor='hand2',
                  command=lambda: self._ajustar_escala(+0.1)
                  ).pack(side='left', padx=(4, 8))

        tk.Button(scale_frame, text='Reset', bg='#313244', fg=FG_DIM,
                  relief='flat', font=('Segoe UI', 9), cursor='hand2',
                  command=lambda: self._ajustar_escala(0, reset=True)
                  ).pack(side='left')

        self.lbl_dims = lbl('dx=—  w=—  h=—', fg=FG_DIM, font=('Consolas', 8))

        sep()

        title('LEGENDA')
        for cor, txt in [
            ('#00d23c', '■ Verde solido   = LV confirmado'),
            ('#ff3232', '■ Verm. solido   = OC confirmado'),
            ('#00aa30', '□ Verde tracejado = LV automatico'),
            ('#cc2020', '□ Verm. tracejado = OC automatico'),
            ('#646464', '□ Cinza          = vaga existente'),
            ('#ffc800', '— Amarelo        = linha guia'),
        ]:
            tk.Label(p, text=f'  {txt}', bg=BG_PANEL, fg=cor,
                     font=('Segoe UI', 9), anchor='w').pack(fill='x', padx=14, pady=0)
        sep()

        title('ESTATISTICAS')
        self.lbl_exist  = lbl('Existentes: —')
        self.lbl_novas  = lbl('Novas vagas: —')
        self.lbl_conf   = lbl('Confirmadas: —')
        self.lbl_n_oc   = lbl('Auto OC: —',  fg='#ff5555')
        self.lbl_n_lv   = lbl('Auto LV: —',  fg='#55ff55')
        sep()

        title('VAGA SELECIONADA')
        lbl('Clique numa vaga nova para ver\ndetalhes e confirmar/inverter',
            fg=FG_DIM, font=('Segoe UI', 9))
        self.txt_det = tk.Text(p, bg='#11111b', fg=FG, font=('Consolas', 9),
                               height=8, relief='flat', wrap='word',
                               state='disabled', padx=8, pady=6)
        self.txt_det.pack(fill='x', padx=14, pady=(4, 10))

    def _build_statusbar(self):
        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                 bg='#12121c', fg=FG_DIM, font=('Segoe UI', 9),
                 anchor='w', padx=10).pack(fill='x', side='bottom')

    # ── Abrir imagem ─────────────────────────────────────────────────────────
    def abrir(self):
        # Preferir pasta cortada; fallback para dataset original
        init = os.path.join(DATASET_G100, 'test')
        if not os.path.isdir(init):
            init = os.path.join(DATASET, 'test')
        path = filedialog.askopenfilename(
            title='Abrir imagem G100 (cortada)',
            initialdir=init,
            filetypes=[('Imagens', '*.jpg *.jpeg *.png'), ('Todos', '*.*')]
        )
        if not path:
            return
        fn = os.path.basename(path)
        info = self.coco_lut.get(fn)
        if info is None:
            messagebox.showwarning('Imagem nao encontrada',
                f'"{fn}" nao esta no COCO.\n'
                'Utilize uma imagem de dataset_parkinglot_g100_crop/test.')
            return
        anns = info['anns']
        if len(anns) != 100:
            messagebox.showwarning('Nao e G100',
                f'Esta imagem tem {len(anns)} vagas anotadas.\n'
                'O anotador e especifico para G100 (100 vagas).')
            return

        # Usar caminho real da imagem (cortada se disponivel)
        actual_path = os.path.join(info['img_dir'], fn)
        if not os.path.exists(actual_path):
            actual_path = path
        self.img_path = actual_path
        # Limpar estado anterior
        self.exist_spots.clear()
        self.novo_spots.clear()
        self.guias.clear()
        self.sel_idx = None

        # Carregar imagem
        raw = imread(actual_path)
        if raw.ndim == 2:
            raw = np.stack([raw]*3, axis=-1)
        elif raw.shape[2] == 4:
            raw = raw[:,:,:3]
        if raw.dtype != np.uint8:
            raw = (raw*255).clip(0,255).astype(np.uint8)
        self.img_pil = Image.fromarray(raw)
        self.img_h, self.img_w = raw.shape[:2]

        # Pre-processar para classificacao (background)
        self._status('A pre-processar imagem...')
        self.root.update()
        self.gray_img = rgb2gray(imread(actual_path))
        self.pp_mg, self.pp_gm, self.pp_me = processar(self.gray_img)

        # Carregar vagas existentes
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt = 'LV' if ann['category_id'] == CAT_LIVRE else 'OC'
            self.exist_spots.append({'x':int(x),'y':int(y),'w':int(w),'h':int(h),'gt':gt})

        # Calcular modelo de perspectiva desta imagem
        self._calibrar_perspectiva(anns)

        self._rebuild_display()
        self.root.after(50, self._zoom_fit_delayed)
        self._update_stats()
        self._status(f'Imagem carregada: {fn}  |  {len(anns)} vagas existentes  |  '
                     f'Modo: Linha — clique+arraste para gerar vagas')

    def _calibrar_perspectiva(self, anns):
        """Ajusta modelo de perspectiva aos dados desta imagem."""
        from collections import defaultdict
        rows = defaultdict(list)
        for ann in anns:
            y_key = int(ann['bbox'][1] // 20) * 20
            rows[y_key].append(ann)

        ys, dxs, ws, hs = [], [], [], []
        for y_key in sorted(rows.keys()):
            row = rows[y_key]
            if len(row) < 3:
                continue
            ys_r  = [a['bbox'][1] for a in row]
            xs_r  = sorted([a['bbox'][0] for a in row])
            ws_r  = [a['bbox'][2] for a in row]
            hs_r  = [a['bbox'][3] for a in row]
            diffs = [xs_r[i+1]-xs_r[i] for i in range(len(xs_r)-1)]
            if not diffs:
                continue
            ys.append(np.mean(ys_r))
            dxs.append(np.median(diffs))
            ws.append(np.mean(ws_r))
            hs.append(np.mean(hs_r))

        if len(ys) >= 2:
            a_dx, b_dx = np.polyfit(ys, dxs, 1)
            a_w,  b_w  = np.polyfit(ys, ws,  1)
            a_h,  b_h  = np.polyfit(ys, hs,  1)
            self.persp = {'a_dx': a_dx, 'b_dx': b_dx,
                          'a_w': a_w,  'b_w': b_w,
                          'a_h': a_h,  'b_h': b_h}
        else:
            self.persp = _PERSP  # fallback global

    # ── Construir imagem display ──────────────────────────────────────────────
    def _rebuild_display(self):
        if self.img_pil is None:
            return
        base    = self.img_pil.convert('RGBA')
        overlay = Image.new('RGBA', base.size, (0,0,0,0))
        draw    = ImageDraw.Draw(overlay)

        # Vagas existentes (cinza translucido)
        for s in self.exist_spots:
            x,y,w,h = s['x'],s['y'],s['w'],s['h']
            fill = RGBA_EXIST_LV if s['gt']=='LV' else RGBA_EXIST_OC
            draw.rectangle([x,y,x+w,y+h], fill=fill)
            for b in range(2):
                draw.rectangle([x-b,y-b,x+w+b,y+h+b], outline=BRD_EXIST)

        # Novas vagas
        for i, s in enumerate(self.novo_spots):
            x,y,w,h = s['x'],s['y'],s['w'],s['h']
            confirmed = s['confirmed']
            lv = s['label'] == 'LV'

            if confirmed:
                fill   = RGBA_LV_CONF if lv else RGBA_OC_CONF
                border = BRD_LV_CONF  if lv else BRD_OC_CONF
                bw     = 3
            else:
                fill   = RGBA_LV_AUTO if lv else RGBA_OC_AUTO
                border = BRD_LV_AUTO  if lv else BRD_OC_AUTO
                bw     = 2

            draw.rectangle([x,y,x+w,y+h], fill=fill)
            # Borda: tracejada para auto, solida para confirmado
            if confirmed:
                for b in range(bw):
                    draw.rectangle([x-b,y-b,x+w+b,y+h+b], outline=border)
            else:
                # Simular tracejado desenhando partes da borda
                step = 5
                for cx in range(x, x+w, step*2):
                    draw.line([(cx,y),(min(cx+step,x+w),y)], fill=border, width=2)
                    draw.line([(cx,y+h),(min(cx+step,x+w),y+h)], fill=border, width=2)
                for cy in range(y, y+h, step*2):
                    draw.line([(x,cy),(x,min(cy+step,y+h))], fill=border, width=2)
                    draw.line([(x+w,cy),(x+w,min(cy+step,y+h))], fill=border, width=2)

            # Highlight selecionado
            if i == self.sel_idx:
                for b in range(4):
                    draw.rectangle([x-b,y-b,x+w+b,y+h+b], outline=BRD_SELECT)

        # Vaga pendente (modo Ponto) — rectangulo rodado a amarelo brilhante
        if self.placing is not None:
            p = self.placing
            corners = self._box_corners(p['cx'], p['cy'], p['w'], p['h'], p['rot'])
            pts = [(int(round(x)), int(round(y))) for x, y in corners]
            # Preenchimento translucido
            draw.polygon(pts, fill=(255, 220, 0, 60))
            # Borda dupla (brilhante)
            for offset in (0, 1):
                pts_off = [(x+offset, y+offset) for x, y in pts]
                draw.line(pts_off + [pts_off[0]], fill=(255, 220, 0, 230), width=2)
            # Cruz no centro
            cx, cy = int(round(p['cx'])), int(round(p['cy']))
            for arm in range(1, 8):
                draw.ellipse([cx-arm, cy-arm, cx+arm, cy+arm],
                             outline=(255, 220, 0, max(20, 230-arm*30)))
            draw.line([(cx-8, cy), (cx+8, cy)], fill=(255, 220, 0, 230), width=2)
            draw.line([(cx, cy-8), (cx, cy+8)], fill=(255, 220, 0, 230), width=2)
            # AABB a tracejado branco (o que sera classificado)
            bx, by, bw, bh = self._placing_aabb()
            step = 6
            for sx in range(bx, bx+bw, step*2):
                draw.line([(sx, by), (min(sx+step, bx+bw), by)],
                          fill=(255,255,255,120), width=1)
                draw.line([(sx, by+bh), (min(sx+step, bx+bw), by+bh)],
                          fill=(255,255,255,120), width=1)
            for sy in range(by, by+bh, step*2):
                draw.line([(bx, sy), (bx, min(sy+step, by+bh))],
                          fill=(255,255,255,120), width=1)
                draw.line([(bx+bw, sy), (bx+bw, min(sy+step, by+bh))],
                          fill=(255,255,255,120), width=1)

        # Guias (linhas amarelas — diagonais reais)
        for g in self.guias:
            draw.line([(g['x1'], g['y1']), (g['x2'], g['y2'])],
                      fill=RGBA_GUIDE, width=2)
            # Extremos com ponto
            for xp, yp in ((g['x1'], g['y1']), (g['x2'], g['y2'])):
                draw.ellipse([xp-4, yp-4, xp+4, yp+4], fill=RGBA_GUIDE)

        self.img_display = Image.alpha_composite(base, overlay).convert('RGB')
        self._render_canvas()

    def _render_canvas(self):
        if self.img_display is None:
            return
        W0,H0 = self.img_display.size
        Wd = max(1, int(W0*self.scale))
        Hd = max(1, int(H0*self.scale))
        scaled = self.img_display.resize((Wd, Hd), Image.LANCZOS)
        self.photo_tk = ImageTk.PhotoImage(scaled)
        self.canvas.delete('all')
        self.canvas.create_image(0,0,anchor='nw',image=self.photo_tk)
        if self._drawing and self._guide_id is not None:
            pass  # guide redrawn in on_drag
        self.canvas.configure(scrollregion=(0,0,Wd,Hd))

    # ── Canvas events ─────────────────────────────────────────────────────────
    def _canvas_to_img(self, cx, cy):
        """Converter coords canvas -> imagem original."""
        return cx/self.scale, cy/self.scale

    def _on_press(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix, iy = self._canvas_to_img(cx, cy)

        if self.mode.get() == 'linha':
            if self.img_pil is None:
                return
            self._drawing    = True
            self._draw_start = (ix, iy)
            self._guide_id   = None

        elif self.mode.get() == 'ponto':
            if self.img_pil is None or self.pp_mg is None:
                return
            self._colocar_ponto(ix, iy)

        elif self.mode.get() == 'confirmar':
            self._click_spot(ix, iy)

    def _on_drag(self, event):
        if not self._drawing:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix, iy = self._canvas_to_img(cx, cy)
        sx, sy = self._draw_start

        # Mostrar guia temporaria — linha diagonal real
        sx_c = sx * self.scale
        sy_c = sy * self.scale
        ix_c = ix * self.scale
        iy_c = iy * self.scale

        if self._guide_id is not None:
            self.canvas.delete(self._guide_id)
        self._guide_id = self.canvas.create_line(
            sx_c, sy_c, ix_c, iy_c,
            fill='#ffc800', width=2, dash=(6, 3))

    def _on_release(self, event):
        if not self._drawing:
            return
        self._drawing = False
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix, iy = self._canvas_to_img(cx, cy)
        sx, sy = self._draw_start

        if self._guide_id is not None:
            self.canvas.delete(self._guide_id)
            self._guide_id = None

        # Ignorar cliques sem arrastar (< 10px Euclidiana)
        dist = np.sqrt((ix - sx)**2 + (iy - sy)**2)
        if dist < 10:
            return

        # Registar guia — preserva direcao (inicio → fim do arrastar)
        # Garantir que x1 <= x2 para consistencia na geracao de vagas
        if sx <= ix:
            guia = {'x1': int(sx), 'y1': int(sy),
                    'x2': int(ix), 'y2': int(iy)}
        else:
            guia = {'x1': int(ix), 'y1': int(iy),
                    'x2': int(sx), 'y2': int(sy)}
        self.guias.append(guia)

        # Gerar e classificar vagas
        self._gerar_vagas_guia(guia)
        self._rebuild_display()
        self._update_stats()
        self._status(f'Linha adicionada: y~{int((sy+iy)/2)}  '
                     f'| {len(self.novo_spots)} novas vagas no total')

    def _on_scroll(self, event):
        if event.state & 0x0004:
            self.zoom(1.1 if event.delta > 0 else 0.909)
        else:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')

    # ── Gerar vagas a partir de uma guia ─────────────────────────────────────
    def _gerar_vagas_guia(self, guia):
        """Gera candidate spots ao longo da guia (diagonal) e classifica."""
        if self.pp_mg is None:
            return

        # Vector unitario ao longo da linha desenhada
        x1, y1 = float(guia['x1']), float(guia['y1'])
        x2, y2 = float(guia['x2']), float(guia['y2'])
        dx_line = x2 - x1
        dy_line = y2 - y1
        length  = np.sqrt(dx_line**2 + dy_line**2)
        if length < 8:
            return
        ux = dx_line / length   # componente x do vector unitario
        uy = dy_line / length   # componente y do vector unitario

        scale    = self.scale_var.get()
        n_antes  = len(self.novo_spots)
        t = 0.0

        while t < length:
            # Ponto central desta vaga ao longo da linha
            cx = x1 + ux * t
            cy = y1 + uy * t

            # Dimensoes perspectiva nesta posicao (ajustadas pela escala manual)
            dx_est, w_est, h_est = perspectiva(cy, self.persp)
            dx_est = max(6, int(round(dx_est * scale)))
            w_est  = max(5, int(round(w_est  * scale)))
            h_est  = max(5, int(round(h_est  * scale)))

            # Bbox centrada no ponto da linha
            sx = max(0, int(round(cx - w_est / 2)))
            sy = max(0, int(round(cy - h_est / 2)))

            # Verificar sobreposicao com vagas existentes (margem 5px)
            sobrepoe = False
            for s in self.exist_spots:
                if not (sx + w_est + 5 <= s['x'] or sx >= s['x'] + s['w'] + 5 or
                        sy + h_est + 5 <= s['y'] or sy >= s['y'] + s['h'] + 5):
                    sobrepoe = True
                    break
            # Verificar sobreposicao com vagas novas ja adicionadas
            if not sobrepoe:
                for s in self.novo_spots:
                    if not (sx + w_est + 2 <= s['x'] or sx >= s['x'] + s['w'] + 2 or
                            sy + h_est + 2 <= s['y'] or sy >= s['y'] + s['h'] + 2):
                        sobrepoe = True
                        break

            if not sobrepoe:
                pred, n_oc = classificar_roi(
                    self.pp_mg, self.pp_gm, self.pp_me,
                    sx, sy, w_est, h_est)
                self.novo_spots.append({
                    'x': sx, 'y': sy, 'w': w_est, 'h': h_est,
                    'pred': pred, 'label': pred,
                    'confirmed': False, 'n_oc': n_oc,
                    'guia_idx': len(self.guias) - 1,
                })
            t += dx_est

        n_gerados = len(self.novo_spots) - n_antes

        # Actualizar painel de dimensoes (valores no ponto medio da guia)
        mid_y = (y1 + y2) / 2
        dx_m, w_m, h_m = perspectiva(mid_y, self.persp)
        self.lbl_dims.configure(
            text=f'dx={int(round(dx_m*scale))}  '
                 f'w={int(round(w_m*scale))}  '
                 f'h={int(round(h_m*scale))}  (escala {scale:.1f}×)')
        print(f'  Guia y~{int(mid_y)}: {n_gerados} vagas geradas '
              f'(dx={int(round(dx_m*scale))} w={int(round(w_m*scale))} '
              f'h={int(round(h_m*scale))} scale={scale:.1f}×)')

    # ── Modo Ponto ────────────────────────────────────────────────────────────
    def _colocar_ponto(self, ix, iy):
        """Coloca/move o rectangulo pendente no ponto (ix, iy)."""
        scale = self.scale_var.get()
        _, w_est, h_est = perspectiva(iy, self.persp)
        w_est = max(5, int(round(w_est * scale)))
        h_est = max(5, int(round(h_est * scale)))
        self.placing = {
            'cx': float(ix), 'cy': float(iy),
            'w': w_est, 'h': h_est,
            'rot': self.rot_deg,
        }
        self._rebuild_display()
        self._atualizar_info_ponto()

    def _nudge_ponto(self, dx, dy):
        """Move o rectangulo pendente pixel a pixel."""
        if self.placing is None:
            return
        self.placing['cx'] += dx
        self.placing['cy'] += dy
        self._rebuild_display()
        self._atualizar_info_ponto()

    def _ajustar_rot_ponto(self, delta, reset=False):
        """Roda o rectangulo pendente delta graus."""
        if reset:
            self.rot_deg = 0.0
        else:
            self.rot_deg = round(self.rot_deg + delta, 1)
        self.lbl_rot.configure(text=f'{self.rot_deg:.1f}°')
        if self.placing is not None:
            self.placing['rot'] = self.rot_deg
            self._rebuild_display()
            self._atualizar_info_ponto()

    def _confirmar_ponto(self):
        """Confirma o rectangulo pendente e classifica a vaga."""
        if self.placing is None:
            return
        p  = self.placing
        bx, by, bw, bh = self._placing_aabb()
        if bw < 2 or bh < 2:
            return
        pred, n_oc = classificar_roi(
            self.pp_mg, self.pp_gm, self.pp_me, bx, by, bw, bh)
        self.novo_spots.append({
            'x': bx, 'y': by, 'w': bw, 'h': bh,
            'pred': pred, 'label': pred,
            'confirmed': False, 'n_oc': n_oc,
            'guia_idx': -1,   # -1 = modo ponto
            'rot': p['rot'],
        })
        self.placing = None
        self._rebuild_display()
        self._update_stats()
        self._status(f'Vaga confirmada — {pred}  ({n_oc}/4 votos OC) '
                     f'| Total: {len(self.novo_spots)}')

    def _cancelar_ponto(self):
        """Cancela o rectangulo pendente sem confirmar."""
        if self.placing is not None:
            self.placing = None
            self._rebuild_display()
            self._status('Vaga cancelada.')

    def _placing_aabb(self):
        """Axis-aligned bounding box do rectangulo rodado pendente."""
        p = self.placing
        cx, cy, w, h, rot = p['cx'], p['cy'], p['w'], p['h'], p['rot']
        corners = self._box_corners(cx, cy, w, h, rot)
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        bx = max(0, int(min(xs)))
        by = max(0, int(min(ys)))
        bw = max(1, int(round(max(xs))) - bx)
        bh = max(1, int(round(max(ys))) - by)
        bw = min(bw, self.img_w - bx)
        bh = min(bh, self.img_h - by)
        return bx, by, bw, bh

    def _box_corners(self, cx, cy, w, h, rot_deg):
        """4 cantos de um rectangulo centrado em (cx,cy) rodado rot_deg graus."""
        cos_t = np.cos(np.radians(rot_deg))
        sin_t = np.sin(np.radians(rot_deg))
        hw, hh = w / 2, h / 2
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return [(cx + x*cos_t - y*sin_t,
                 cy + x*sin_t + y*cos_t) for x, y in local]

    def _atualizar_info_ponto(self):
        """Actualiza o label de informacao do ponto actual."""
        if self.placing is None:
            self.lbl_ponto_info.configure(text='—')
            return
        p = self.placing
        bx, by, bw, bh = self._placing_aabb()
        self.lbl_ponto_info.configure(
            text=f'cx={int(p["cx"])} cy={int(p["cy"])}  '
                 f'{bw}×{bh}px  {p["rot"]:.1f}°')

    # ── Click numa vaga nova ──────────────────────────────────────────────────
    def _click_spot(self, ix, iy):
        # Procurar vaga nova que contenha o ponto
        for i, s in enumerate(self.novo_spots):
            if s['x'] <= ix <= s['x']+s['w'] and s['y'] <= iy <= s['y']+s['h']:
                self.sel_idx = i
                self._toggle_label(i)
                self._rebuild_display()
                self._mostrar_detalhe(i)
                self._update_stats()
                return
        # Click em area vazia: deseleccionar
        self.sel_idx = None
        self._rebuild_display()
        self._limpar_detalhe()

    def _toggle_label(self, idx):
        s = self.novo_spots[idx]
        if not s['confirmed']:
            # Primeira confirmacao: confirmar o label actual
            s['confirmed'] = True
        else:
            # Ja confirmado: inverter label
            s['label'] = 'LV' if s['label'] == 'OC' else 'OC'
        self._update_stats()

    # ── Ajuste de escala das vagas ────────────────────────────────────────────
    def _ajustar_escala(self, delta, reset=False):
        if reset:
            self.scale_var.set(1.0)
        else:
            new = round(max(0.2, min(3.0, self.scale_var.get() + delta)), 1)
            self.scale_var.set(new)
        self.lbl_escala.configure(text=f'{self.scale_var.get():.1f}×')

    # ── Desfazer ─────────────────────────────────────────────────────────────
    def desfazer(self):
        # Se ha vaga pendente em modo ponto, cancelar primeiro
        if self.placing is not None:
            self._cancelar_ponto()
            return
        # Remover ultima vaga de ponto (guia_idx == -1)
        if self.novo_spots and self.novo_spots[-1].get('guia_idx') == -1:
            self.novo_spots.pop()
            if self.sel_idx is not None and self.sel_idx >= len(self.novo_spots):
                self.sel_idx = None
            self._rebuild_display()
            self._update_stats()
            self._limpar_detalhe()
            self._status(f'Ultima vaga (ponto) removida. {len(self.novo_spots)} vagas restantes.')
            return
        # Remover ultima linha guia e as suas vagas
        if not self.guias:
            return
        guia_idx = len(self.guias) - 1
        self.guias.pop()
        self.novo_spots = [s for s in self.novo_spots
                           if s.get('guia_idx') != guia_idx]
        if self.sel_idx is not None and self.sel_idx >= len(self.novo_spots):
            self.sel_idx = None
        self._rebuild_display()
        self._update_stats()
        self._limpar_detalhe()
        self._status(f'Ultima linha removida. {len(self.guias)} guias restantes.')

    # ── Exportar CSV ─────────────────────────────────────────────────────────
    def exportar(self):
        if not self.novo_spots:
            messagebox.showinfo('Sem dados', 'Nao ha vagas novas para exportar.')
            return
        fn  = os.path.basename(self.img_path) if self.img_path else 'unknown.jpg'
        out = os.path.join(EXPORT_DIR,
                           fn.replace('.jpg', '_anotacoes.csv')
                              .replace('.jpeg','_anotacoes.csv'))
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['filename','parque','bbox_x','bbox_y','bbox_w','bbox_h',
                        'label','confirmed','pred_auto','n_votos_oc'])
            for s in self.novo_spots:
                w.writerow([fn, 'G100', s['x'], s['y'], s['w'], s['h'],
                             s['label'], int(s['confirmed']),
                             s['pred'], s['n_oc']])
        n_conf = sum(1 for s in self.novo_spots if s['confirmed'])
        messagebox.showinfo('Exportado',
            f'{len(self.novo_spots)} vagas guardadas\n'
            f'({n_conf} confirmadas)\n\n{out}')
        self._status(f'Exportado: {os.path.basename(out)}  '
                     f'({len(self.novo_spots)} vagas)')

    # ── Zoom ─────────────────────────────────────────────────────────────────
    def zoom(self, factor):
        self.scale = max(0.05, min(10.0, self.scale * factor))
        self.lbl_zoom.configure(text=f'{self.scale*100:.0f}%')
        self._render_canvas()

    def zoom_fit(self):
        if self.img_display is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        W0,H0 = self.img_display.size
        self.scale = min(cw/W0, ch/H0, 1.0)
        self.lbl_zoom.configure(text=f'{self.scale*100:.0f}%')
        self._render_canvas()

    def _zoom_fit_delayed(self):
        self._render_canvas()
        self.zoom_fit()

    # ── Stats ─────────────────────────────────────────────────────────────────
    def _update_stats(self):
        n_novo = len(self.novo_spots)
        n_conf = sum(1 for s in self.novo_spots if s['confirmed'])
        n_oc   = sum(1 for s in self.novo_spots if s['label'] == 'OC')
        n_lv   = sum(1 for s in self.novo_spots if s['label'] == 'LV')
        self.lbl_exist.configure( text=f'Existentes: {len(self.exist_spots)}')
        self.lbl_novas.configure( text=f'Novas vagas: {n_novo}')
        self.lbl_conf.configure(  text=f'Confirmadas: {n_conf} / {n_novo}')
        self.lbl_n_oc.configure(  text=f'Label OC: {n_oc}')
        self.lbl_n_lv.configure(  text=f'Label LV: {n_lv}')

    # ── Detalhe ───────────────────────────────────────────────────────────────
    def _mostrar_detalhe(self, idx):
        s = self.novo_spots[idx]
        self.txt_det.configure(state='normal')
        self.txt_det.delete('1.0', 'end')
        pred_cor = '#ff5555' if s['pred'] == 'OC' else '#55ff55'
        lbl_cor  = '#ff5555' if s['label'] == 'OC' else '#55ff55'
        st_cor   = '#aaaaaa' if not s['confirmed'] else ('#ff5555' if s['label']=='OC' else '#55ff55')

        self.txt_det.insert('end', f'Vaga #{idx+1}\n')
        self.txt_det.insert('end', f'bbox: ({s["x"]},{s["y"]}) {s["w"]}x{s["h"]}px\n')
        self.txt_det.insert('end', f'Pred. auto: {s["pred"]} ({s["n_oc"]}/4 votos)\n')
        self.txt_det.insert('end', f'Label atual: {s["label"]}\n')
        self.txt_det.insert('end', f'Confirmado: {"Sim" if s["confirmed"] else "Nao"}\n')
        self.txt_det.insert('end', '\n1 click = confirmar\n2 click = inverter label')
        self.txt_det.configure(state='disabled')

    def _limpar_detalhe(self):
        self.txt_det.configure(state='normal')
        self.txt_det.delete('1.0', 'end')
        self.txt_det.insert('end', 'Clique numa vaga nova\npara ver detalhes.')
        self.txt_det.configure(state='disabled')

    def _status(self, msg):
        self.status_var.set(f'  {msg}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    try:
        root.tk.call('tk', 'scaling', 1.25)
    except Exception:
        pass
    app = AnotadorApp(root)
    root.mainloop()
