from dataclasses import dataclass, field
from pathlib import Path
import os
import numpy as np

@dataclass
class CalibrationConfig:
    board_corners: tuple[int, int]
    square_size_mm: float
    device: str
    colab: bool
    use_synthetic: bool
    alpha: float
    root_dir: Path = field(init=False)

    def __post_init__(self):
        if self.colab:
            from google.colab import drive
            drive.mount("/content/drive")
            self.root_dir = Path("/content/drive/MyDrive/CI1026-Visão Computacional/TA02 - Calibração de Imagem/calibracao")
        else:
            self.root_dir = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Experiment3DConfig:
    """Geometry used to validate 3D points projected into the image."""

    cube_origin_squares: tuple[float, float] = (3.0, 1.0)
    cube_side_squares: float = 3.0
    axis_length_squares: float = 3.0
    cube_depth_direction: float = -1.0


@dataclass
class SyntheticConfig:
    width : int
    height : int
    K : np.ndarray
    distortion : np.ndarray
    stereo_distance : float
    pixels_per_square : int


@dataclass
class DatasetPaths:

    def __init__(self, root_dir, use_synthetic, device : str = "") -> None:
        suffix = "synthetic" if use_synthetic else device
        self.base = Path(root_dir) / suffix
        self.input = self.base / "input"
        self.calibration = self.input / "calibration"
        self.test = self.input / "test"
        self.stereo_left = self.input / "estereo" / "esquerda"
        self.stereo_right = self.input / "estereo" / "direita"
        self.output = self.base / "output"
        self.figures = self.output / "figures"
        self.corrected_images = self.output / "corrected_images"
        self.metrics = self.output / "metrics"
        self.create_directories()

    def create_directories(self):
        for path in [self.calibration, self.test, self.stereo_left, self.stereo_right,
                     self.figures, self.corrected_images, self.metrics]:
            os.makedirs(path, exist_ok=True)



## ===== CONFIGURAÇÕES =====
#CANTOS = (9, 4)      # cantos INTERNOS do tabuleiro (colunas, linhas)
#QUAD_MM = 90      # lado de um quadrado, em milímetros
#
## True  -> gera e usa fotos sintéticas (K e distorção conhecidos) para validar o código
## False -> usa as fotos reais colocadas na pasta data/{device}
#USAR_SINTETICAS = False
#DEVICE = 'android'
#
#if NO_COLAB:
#    RAIZ = "/content/drive/MyDrive/CI1026-Visão Computacional/TA02 - Calibração de Imagem/calibracao"
#else:
#    RAIZ = os.path.abspath("data")
#
## sintéticas e reais ficam separadas, cada uma com a mesma organização
#BASE = os.path.join(RAIZ, "synthetic" if USAR_SINTETICAS else f"{DEVICE}")
#
#
## ---------- ENTRADA: onde o código LÊ as fotos ----------
#ENTRADA     = os.path.join(BASE, "input")
#PASTA_CALIB = os.path.join(ENTRADA, "calibration")      # fotos do tabuleiro
#PASTA_TESTE = os.path.join(ENTRADA, "test")           # fotos para o experimento 3D->2D
#PASTA_ESQ   = os.path.join(ENTRADA, "estereo", "esquerda")   # pares estéreo (mesmo nome nas duas)
#PASTA_DIR   = os.path.join(ENTRADA, "estereo", "direita")
#
## ---------- SAÍDA: onde o código GRAVA os resultados ----------
#SAIDA          = os.path.join(BASE, "output")
#PASTA_FIGURAS  = os.path.join(SAIDA, "figures")               # figuras do artigo
#PASTA_CORRIG   = os.path.join(SAIDA, "corrected_images") # todas as fotos corrigidas
#PASTA_DADOS    = os.path.join(SAIDA, "metrics")                 # matrizes, json, tabelas LaTeX
#
#for p in [PASTA_CALIB, PASTA_TESTE, #PASTA_ESQ, PASTA_DIR,
#          PASTA_FIGURAS, PASTA_CORRIG, PASTA_DADOS]:
#    os.makedirs(p, exist_ok=True)
#
#EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif")
#RESULTADOS = {}                         # tudo que vai para o relatório
#
#def salvar_fig(nome):
#    plt.savefig(os.path.join(PASTA_FIGURAS, nome), dpi=150, bbox_inches="tight")
#
#print("Cantos internos:", CANTOS, "| quadrado:", QUAD_MM, "mm")
#print("Lendo de :", ENTRADA)
#print("Gravando em:", SAIDA)
