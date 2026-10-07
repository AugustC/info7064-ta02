from dataclasses import dataclass

import cv2
import numpy as np

from calibration.config import CalibrationConfig


CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)

@dataclass
class CalibrationResults:
    rms: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation_vectors: list[np.ndarray]
    translation_vectors: list[np.ndarray]
    reprojection_errors: np.ndarray
    image_size: tuple[int, int]
    fov_x: float
    fov_y: float
    std_int: float
    std_ext: float


def create_object_points(corners, square_size_mm):
    object_points = np.zeros((corners[0] * corners[1], 3), np.float32)
    object_points[:, :2] = (
        np.mgrid[0:corners[0], 0:corners[1]].T.reshape(-1, 2) * square_size_mm
    )
    return object_points


def detect_corners(img, corners, max_side=1600):
    # cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # esc = min(1.0, lado_max / max(cinza.shape))
    # peq = cv2.resize(cinza, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) if esc < 1 else cinza
    # flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    # ok, c = cv2.findChessboardCorners(peq, CANTOS, flags)
    # if not ok:
    #     ok, c = cv2.findChessboardCornersSB(peq, CANTOS, cv2.CALIB_CB_EXHAUSTIVE)
    # if not ok:
    #     return None
    # c = (c / esc).astype(np.float32)
    # jan = max(5, int(round(5 / esc)))
    # return cv2.cornerSubPix(cinza, c, (jan, jan), (-1, -1), CRIT)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = min(1.0, max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, detected = cv2.findChessboardCorners(small, corners, flags)
    if not ok:
        ok, detected = cv2.findChessboardCornersSB(small, corners, cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return None
    detected = (detected / scale).astype(np.float32)
    window = max(5, int(round(5 / scale)))
    return cv2.cornerSubPix(gray, detected, (window, window), (-1, -1), CRIT)


def collect_calibration_points(image_paths, corners, square_size_mm):
    # obj_pts, img_pts, usadas = [], [], []
    # for p in arquivos:
    #     img = carregar(p)
    #     if img.shape[:2][::-1] != TAM:
    #         print("  tamanho diferente, ignorada:", os.path.basename(p)); continue
    #     c = detectar(img)
    #     if c is None:
    #         print("  tabuleiro NÃO encontrado:", os.path.basename(p)); continue
    #     obj_pts.append(OBJP); img_pts.append(c); usadas.append(p)
    #
    #   NOTE: Pega o tamanho da primeira image ao invés de usar uma constante global
    object_points = create_object_points(corners, square_size_mm)
    object_points_list, image_points_list, used_paths = [], [], []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print("  imagem não pôde ser lida, ignorada:", path)
            continue
        current_size = image.shape[:2][::-1]
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            print("  tamanho diferente, ignorada:", path)
            continue
        detected = detect_corners(image, corners)
        if detected is None:
            print("  tabuleiro NÃO encontrado:", path)
            continue
        object_points_list.append(object_points)
        image_points_list.append(detected)
        used_paths.append(path)

    return object_points_list, image_points_list, used_paths, image_size


def calibrate_camera(obj_pts, img_pts, image_size):
    # rms, K, dist, rvecs, tvecs, std_int, std_ext, err_views = cv2.calibrateCameraExtended(
    #     obj_pts, img_pts, TAM, None, None, criteria=CRIT)
    # dist = dist.ravel()
    # std_int = std_int.ravel()
    rms, camera_matrix, distortion, rotation_vectors, translation_vectors, std_int, std_ext, *_ = cv2.calibrateCameraExtended(
        obj_pts,
        img_pts,
        image_size,
        None,
        None,
        criteria=CRIT,
    )
    distortion = distortion.ravel()
    errors = calculate_reprojection_error(
        obj_pts,
        img_pts,
        rotation_vectors,
        translation_vectors,
        camera_matrix,
        distortion,
    )
    fov_x, fov_y = calculate_field_of_view(camera_matrix, image_size)
    return CalibrationResults(
        rms=float(rms),
        camera_matrix=camera_matrix,
        distortion=distortion,
        rotation_vectors=rotation_vectors,
        translation_vectors=translation_vectors,
        reprojection_errors=errors,
        image_size=image_size,
        fov_x=fov_x,
        fov_y=fov_y,
        std_int=std_int,
        std_ext=std_ext
    )


def calculate_reprojection_error(obj_pts, img_pts, rvecs, tvecs, K, dist):
    # erros = []
    # for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
    #     proj, _ = cv2.projectPoints(o, r, t, K, dist)
    #     erros.append(np.linalg.norm(proj.reshape(-1, 2) - i.reshape(-1, 2), axis=1).mean())
    # erros = np.array(erros)
    errors = []
    for object_points, image_points, rvec, tvec in zip(
        obj_pts, img_pts, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, K, dist
        )
        errors.append(
            np.linalg.norm(
                projected.reshape(-1, 2) - image_points.reshape(-1, 2),
                axis=1,
            ).mean()
        )
    return np.array(errors)


def calculate_field_of_view(K, image_size):
    # fov_x = 2 * np.degrees(np.arctan(TAM[0] / (2 * K[0, 0])))
    # fov_y = 2 * np.degrees(np.arctan(TAM[1] / (2 * K[1, 1])))
    fov_x = 2 * np.degrees(np.arctan(image_size[0] / (2 * K[0, 0])))
    fov_y = 2 * np.degrees(np.arctan(image_size[1] / (2 * K[1, 1])))
    return float(fov_x), float(fov_y)


def create_undistortion_maps(K, dist, image_size, config: CalibrationConfig | None = None):
    # ALPHA = 1
    # K_novo, roi = cv2.getOptimalNewCameraMatrix(K, dist, TAM, alpha=ALPHA, newImgSize=TAM)
    # mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_novo, TAM, cv2.CV_32FC1)
    alpha = config.alpha if config is not None else 1
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        K, dist, image_size, alpha=alpha, newImgSize=image_size
    )
    return cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        new_camera_matrix,
        image_size,
        cv2.CV_32FC1,
    )


def undistort_image(img, maps):
    # def corrigir(img):
    #     return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    map_x, map_y = maps
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)


def calculate_straightness_error(
    corners,
    config: CalibrationConfig,
):
    # def retidao(c):
    #     c = c.reshape(CANTOS[1], CANTOS[0], 2); d = []
    #     for linha in list(c) + list(c.transpose(1, 0, 2)):
    #         vx, vy, x0, y0 = cv2.fitLine(linha.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    #         d.append(np.abs((linha[:,0]-x0)*vy - (linha[:,1]-y0)*vx).mean())
    #     return float(np.mean(d))
    points = corners.reshape(
        config.board_corners[1],
        config.board_corners[0],
        2,
    )
    deviations = []

    for line in list(points) + list(points.transpose(1, 0, 2)):
        vx, vy, x0, y0 = cv2.fitLine(
            line.astype(np.float32),
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).ravel()

        deviations.append(
            np.abs(
                (line[:, 0] - x0) * vy
                - (line[:, 1] - y0) * vx
            ).mean()
        )

    return float(np.mean(deviations))


def estimate_board_pose(object_points, image_points, K, dist):
    # ok, r, t = cv2.solvePnP(OBJP[ids_ext], c[ids_ext], K, dist, flags=cv2.SOLVEPNP_IPPE)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    return ok, rvec, tvec


def project_board_points(object_points, rvec, tvec, K, dist):
    # proj = cv2.projectPoints(OBJP, r, t, K, dist)[0].reshape(-1, 2)
    return cv2.projectPoints(object_points, rvec, tvec, K, dist)[0].reshape(
        -1, 2
    )


# ===== PONTOS 3D DO TABULEIRO (Z = 0) =====
# OBJP = np.zeros((CANTOS[0] * CANTOS[1], 3), np.float32)
# OBJP[:, :2] = np.mgrid[0:CANTOS[0], 0:CANTOS[1]].T.reshape(-1, 2) * QUAD_MM
#
# CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
#
# obj_pts, img_pts, usadas = [], [], []
# for p in arquivos:
#     img = carregar(p)
#     if img.shape[:2][::-1] != TAM:
#         print("  tamanho diferente, ignorada:", os.path.basename(p)); continue
#     c = detectar(img)
#     if c is None:
#         print("  tabuleiro NÃO encontrado:", os.path.basename(p)); continue
#     obj_pts.append(OBJP); img_pts.append(c); usadas.append(p)
