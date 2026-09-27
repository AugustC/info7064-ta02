import numpy as np
from calibration.config import CalibrationConfig, DatasetPaths
import cv2
import os
from .config import SyntheticConfig

#W = 1600
#H = 1200
#K = np.array([[1400.0, 0, 810.0],
#                   [0, 1395.0, 590.0],
#                   [0, 0, 1.0]])
#DIST = np.array([-0.28, 0.12, 0.001, -0.0015, -0.02])
#DIST_REAL = np.array([-0.28, 0.12, 0.001, -0.0015, -0.02])   # k1 k2 p1 p2 k3
#
#DIST_STEREO = 120.0                                   # distância entre as câmeras estéreo (mm)
#PX = 60                                             # pixels por quadrado na textura

def create_board_texture(corners, px):
    nx, ny = corners[0] + 1, corners[1] + 1
    plano = np.full(((ny + 2) * px, (nx + 2) * px), 245, np.uint8)
    for i in range(ny):
        for j in range(nx):
            if (i + j) % 2 == 0:
                plano[(i+1)*px:(i+2)*px, (j+1)*px:(j+2)*px] = 20
    return plano

def create_camera_rays(synthetic_config : SyntheticConfig):
    u, v = np.meshgrid(np.arange(synthetic_config.width, dtype=np.float32), np.arange(synthetic_config.height, dtype=np.float32))
    pixels = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pixels, synthetic_config.K, synthetic_config.distortion).reshape(-1, 2)
    return np.hstack([norm, np.ones((len(norm), 1))])

def render_board(rvec, tvec, calibration_config : CalibrationConfig, synthetic_config : SyntheticConfig, rng):
    # desenha o tabuleiro visto de uma posição/orientação da câmera
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    n = R[:, 2]                     # normal do plano do tabuleiro
    rays = create_camera_rays(synthetic_config)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (n @ t) / (rays @ n)   # onde cada raio encontra o plano
    P = rays * s[:, None]
    Pw = (P - t) @ R                # ponto no sistema do tabuleiro
    px = synthetic_config.pixels_per_square
    W, H = synthetic_config.width, synthetic_config.height
    scale = calibration_config.square_size_mm / px                                  # milímetros por pixel da textura
    mx = (Pw[:, 0] / scale + px).astype(np.float32).reshape(H, W)
    my = (Pw[:, 1] / scale + px).astype(np.float32).reshape(H, W)
    plane = create_board_texture(calibration_config.board_corners, px)
    img = cv2.remap(plane, mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=160)
    img[~(s.reshape(H, W) > 0)] = 160
    img = cv2.GaussianBlur(img, (3, 3), 0.6)
    return np.clip(img.astype(np.float32) + rng.normal(0, 2.5, img.shape),
                   0, 255).astype(np.uint8)

def pose_fits_in_frame(rvec,
                       tvec,
                       calibration_config : CalibrationConfig,
                       synthetic_config : SyntheticConfig,
                       margin=160):
    corners = calibration_config.board_corners
    square_size_mm = calibration_config.square_size_mm
    height, width = synthetic_config.height, synthetic_config.width
    K = synthetic_config.K
    distortion = synthetic_config.distortion

    ox, oy = np.meshgrid(np.arange(corners[0]), np.arange(corners[1]))
    pts = np.stack([ox.ravel() * square_size_mm, oy.ravel() * square_size_mm, np.zeros(ox.size)], 1)
    p = cv2.projectPoints(pts, rvec, tvec, K, distortion)[0].reshape(-1, 2)
    return (p[:,0].min() > margin and p[:,0].max() < width - margin and
            p[:,1].min() > margin and p[:,1].max() < height - margin)

def sample_pose(rng):
    rvec = np.array([rng.uniform(-.5, .5), rng.uniform(-.55, .55),
                     rng.uniform(-.4, .4)]).reshape(3, 1)
    tvec = np.array([rng.uniform(-480, 480), rng.uniform(-480, 480),
                     rng.uniform(1120, 1450)]).reshape(3, 1)
    return rvec, tvec

def generate_synthetic_images(num_calib,
                              num_test,
                              num_stereo,
                              dataset_paths : DatasetPaths,
                              calibration_config : CalibrationConfig,
                              synthetic_config : SyntheticConfig):
    num_calib, num_test, num_stereo = 25, 3, 12
    poses_teste = []
    feitas = 0
    rng = np.random.default_rng(7)
    while feitas < num_calib + num_test:
        rvec, tvec = sample_pose(rng)
        if not pose_fits_in_frame(rvec, tvec, calibration_config, synthetic_config):
            continue
        img = render_board(rvec, tvec, calibration_config, synthetic_config, rng)
        if feitas < num_calib:
            cv2.imwrite(os.path.join(dataset_paths.calibration, f"sint_{feitas:02d}.png"), img)
        else:
            cv2.imwrite(os.path.join(dataset_paths.test, f"sint_teste_{feitas-num_calib:02d}.png"), img)
            poses_teste.append((rvec.ravel(), tvec.ravel()))
        feitas += 1
        print(f"Geradas {feitas} imagens sintéticas de calibração e teste", end="\r")

    # pares estéreo: câmera direita deslocada DIST_STEREO mm no eixo x da esquerda
    feitas = 0
    while feitas < num_stereo:
        rvec, tvec = sample_pose(rng)
        tvec_d = tvec - np.array([[synthetic_config.stereo_distance], [0], [0]])
        if not (pose_fits_in_frame(rvec, tvec, calibration_config, synthetic_config) and pose_fits_in_frame(rvec, tvec_d, calibration_config, synthetic_config)):
            continue
        cv2.imwrite(os.path.join(dataset_paths.stereo_left, f"par_{feitas:02d}.png"), render_board(rvec, tvec, calibration_config, synthetic_config, rng))
        cv2.imwrite(os.path.join(dataset_paths.stereo_right, f"par_{feitas:02d}.png"), render_board(rvec, tvec_d, calibration_config, synthetic_config, rng))
        feitas += 1

    np.savez(os.path.join(dataset_paths.input, "verdade_sintetica.npz"), K=synthetic_config.K, dist=synthetic_config.distortion,
             base=synthetic_config.stereo_distance, poses_teste=np.array([np.r_[r, t] for r, t in poses_teste]))
    print(f"{num_calib} de calibração, {num_test} de teste, {num_stereo} pares estéreo")



#rng = np.random.default_rng(7)
#
#nx, ny = CANTOS[0] + 1, CANTOS[1] + 1
#PLANO = np.full(((ny + 2) * PX, (nx + 2) * PX), 245, np.uint8)   # margem branca
#for i in range(ny):
#    for j in range(nx):
#        if (i + j) % 2 == 0:
#            PLANO[(i+1)*PX:(i+2)*PX, (j+1)*PX:(j+2)*PX] = 20
#ESC = QUAD_MM / PX                                  # milímetros por pixel da textura
#
## para cada pixel da foto, a direção do raio que sai da câmera
#u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
#pixels = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
#norm = cv2.undistortPoints(pixels, K_REAL, DIST_REAL).reshape(-1, 2)
#RAIOS = np.hstack([norm, np.ones((len(norm), 1))])
#
#def render(rvec, tvec):
#    # desenha o tabuleiro visto de uma posição/orientação da câmera
#    R, _ = cv2.Rodrigues(rvec)
#    t = tvec.reshape(3)
#    n = R[:, 2]                     # normal do plano do tabuleiro
#    with np.errstate(divide="ignore", invalid="ignore"):
#        s = (n @ t) / (RAIOS @ n)   # onde cada raio encontra o plano
#    P = RAIOS * s[:, None]
#    Pw = (P - t) @ R                # ponto no sistema do tabuleiro
#    mx = (Pw[:, 0] / ESC + PX).astype(np.float32).reshape(H, W)
#    my = (Pw[:, 1] / ESC + PX).astype(np.float32).reshape(H, W)
#    img = cv2.remap(PLANO, mx, my, cv2.INTER_LINEAR,
#                    borderMode=cv2.BORDER_CONSTANT, borderValue=160)
#    img[~(s.reshape(H, W) > 0)] = 160
#    img = cv2.GaussianBlur(img, (3, 3), 0.6)
#    return np.clip(img.astype(np.float32) + rng.normal(0, 2.5, img.shape),
#                   0, 255).astype(np.uint8)
#
#def cabe_na_foto(rvec, tvec, margem=45):
#    ox, oy = np.meshgrid(np.arange(CANTOS[0]), np.arange(CANTOS[1]))
#    pts = np.stack([ox.ravel() * QUAD_MM, oy.ravel() * QUAD_MM, np.zeros(ox.size)], 1)
#    p = cv2.projectPoints(pts, rvec, tvec, K_REAL, DIST_REAL)[0].reshape(-1, 2)
#    return (p[:,0].min() > margem and p[:,0].max() < W - margem and
#            p[:,1].min() > margem and p[:,1].max() < H - margem)
#
#def sorteia_pose():
#    rvec = np.array([rng.uniform(-.5, .5), rng.uniform(-.55, .55),
#                     rng.uniform(-.4, .4)]).reshape(3, 1)
#    tvec = np.array([rng.uniform(-80, 80), rng.uniform(-60, 60),
#                     rng.uniform(380, 750)]).reshape(3, 1)
#    return rvec, tvec
#
#if USAR_SINTETICAS:
#    N_CALIB, N_TESTE, N_ESTEREO = 25, 3, 12
#    poses_teste = []
#    feitas = 0
#    while feitas < N_CALIB + N_TESTE:
#        rvec, tvec = sorteia_pose()
#        if not cabe_na_foto(rvec, tvec):
#            continue
#        img = render(rvec, tvec)
#        if feitas < N_CALIB:
#            cv2.imwrite(os.path.join(PASTA_CALIB, f"sint_{feitas:02d}.png"), img)
#        else:
#            cv2.imwrite(os.path.join(PASTA_TESTE, f"sint_teste_{feitas-N_CALIB:02d}.png"), img)
#            poses_teste.append((rvec.ravel(), tvec.ravel()))
#        feitas += 1
#
#    # pares estéreo: câmera direita deslocada DIST_STEREO mm no eixo x da esquerda
#    feitas = 0
#    while feitas < N_ESTEREO:
#        rvec, tvec = sorteia_pose()
#        tvec_d = tvec - np.array([[DIST_STEREO], [0], [0]])
#        if not (cabe_na_foto(rvec, tvec) and cabe_na_foto(rvec, tvec_d)):
#            continue
#        cv2.imwrite(os.path.join(PASTA_ESQ, f"par_{feitas:02d}.png"), render(rvec, tvec))
#        cv2.imwrite(os.path.join(PASTA_DIR, f"par_{feitas:02d}.png"), render(rvec, tvec_d))
#        feitas += 1
#
#    np.savez(os.path.join(ENTRADA, "verdade_sintetica.npz"), K=K_REAL, dist=DIST_REAL,
#             base=DIST_STEREO, poses_teste=np.array([np.r_[r, t] for r, t in poses_teste]))
#    print(f"{N_CALIB} de calibração, {N_TESTE} de teste, {N_ESTEREO} pares estéreo")
