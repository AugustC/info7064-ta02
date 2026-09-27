from dataclasses import dataclass

import cv2
import numpy as np

from calibration.camera import create_object_points, detect_corners
from calibration.config import CalibrationConfig


@dataclass
class StereoResults:
    rms: float
    rotation: np.ndarray
    translation: np.ndarray
    essential: np.ndarray
    fundamental: np.ndarray
    baseline: float
    angle_degrees: float
    object_points: list[np.ndarray]
    left_points: list[np.ndarray]
    right_points: list[np.ndarray]


def collect_stereo_results(left_paths, right_paths, config: CalibrationConfig):
    # pares_e, pares_d = listar(PASTA_ESQ), listar(PASTA_DIR)
    # o_s, e_s, d_s = [], [], []
    # for pe, pd in zip(pares_e, pares_d):
    #     ce, cd = detectar(carregar(pe)), detectar(carregar(pd))
    #     if ce is not None and cd is not None:
    #         o_s.append(OBJP); e_s.append(ce); d_s.append(cd)
    object_points = create_object_points(
        config.board_corners,
        config.square_size_mm,
    )
    stereo_object_points, left_points, right_points = [], [], []

    for left_path, right_path in zip(left_paths, right_paths):
        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_image is None or right_image is None:
            continue
        left_corners = detect_corners(left_image, config.board_corners)
        right_corners = detect_corners(right_image, config.board_corners)
        if left_corners is not None and right_corners is not None:
            stereo_object_points.append(object_points)
            left_points.append(left_corners)
            right_points.append(right_corners)

    return stereo_object_points, left_points, right_points


def calibrate_stereo(
    object_points,
    left_points,
    right_points,
    K1,
    d1,
    K2,
    d2,
    image_size,
):
    # rms_s, _, _, _, _, R_s, T_s, E_s, F_s = cv2.stereoCalibrate(
    #     o_s[:-1], e_s[:-1], d_s[:-1], K1, d1, K2, d2, TAM,
    #     flags=cv2.CALIB_FIX_INTRINSIC, criteria=CRIT)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )
    rms, _, _, _, _, rotation, translation, essential, fundamental = (
        cv2.stereoCalibrate(
            object_points[:-1],
            left_points[:-1],
            right_points[:-1],
            K1,
            d1,
            K2,
            d2,
            image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=criteria,
        )
    )
    baseline = float(np.linalg.norm(translation))
    angle_degrees = float(
        np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0]))
    )
    return StereoResults(
        rms=float(rms),
        rotation=rotation,
        translation=translation,
        essential=essential,
        fundamental=fundamental,
        baseline=baseline,
        angle_degrees=angle_degrees,
        object_points=object_points,
        left_points=left_points,
        right_points=right_points,
    )


def triangulate_points(left_points, right_points, K1, d1, K2, d2, rotation, translation):
    # ne = cv2.undistortPoints(ce, K1, d1).reshape(-1, 2).T
    # nd = cv2.undistortPoints(cd, K2, d2).reshape(-1, 2).T
    # P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    # P2 = np.hstack([R_s, T_s.reshape(3, 1)])
    # Xh = cv2.triangulatePoints(P1, P2, ne, nd)
    # X = (Xh[:3] / Xh[3]).T
    left_normalized = cv2.undistortPoints(
        left_points, K1, d1
    ).reshape(-1, 2).T
    right_normalized = cv2.undistortPoints(
        right_points, K2, d2
    ).reshape(-1, 2).T
    projection_left = np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_right = np.hstack(
        [rotation, translation.reshape(3, 1)]
    )
    homogeneous = cv2.triangulatePoints(
        projection_left,
        projection_right,
        left_normalized,
        right_normalized,
    )
    return (homogeneous[:3] / homogeneous[3]).T


def compare_triangulation_with_pnp(
    points_3d,
    object_points,
    left_points,
    K,
    dist,
    square_size_mm,
):
    # G = X.reshape(CANTOS[1], CANTOS[0], 3)
    # lados = np.concatenate([np.linalg.norm(np.diff(G, axis=1), axis=2).ravel(),
    #                         np.linalg.norm(np.diff(G, axis=0), axis=2).ravel()])
    # ok, r, t = cv2.solvePnP(OBJP, ce, K1, d1)
    # X_ref = (cv2.Rodrigues(r)[0] @ OBJP.T + t).T
    points_grid = points_3d.reshape(
        int(len(object_points) / (len(object_points) // 2)),
        len(object_points) // int(len(object_points) / (len(object_points) // 2)),
        3,
    )
    sides = np.concatenate(
        [
            np.linalg.norm(np.diff(points_grid, axis=1), axis=2).ravel(),
            np.linalg.norm(np.diff(points_grid, axis=0), axis=2).ravel(),
        ]
    )
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        left_points,
        K,
        dist,
    )
    reference = (cv2.Rodrigues(rvec)[0] @ object_points.T + tvec).T
    difference = np.linalg.norm(points_3d - reference, axis=1)
    return {
        "square_size": (float(sides.mean()), float(sides.std()), square_size_mm),
        "difference": (float(difference.mean()), float(difference.max())),
        "reference": reference,
        "rotation": rvec,
        "translation": tvec,
        "ok": ok,
    }


# ===== CALIBRAÇÃO ESTÉREO =====
# pares_e, pares_d = listar(PASTA_ESQ), listar(PASTA_DIR)
# FAZER_ESTEREO = len(pares_e) > 0 and len(pares_e) == len(pares_d)
# if FAZER_ESTEREO:
#     K1, d1 = K, dist
#     K2, d2 = K, dist
#     o_s, e_s, d_s = [], [], []
#     for pe, pd in zip(pares_e, pares_d):
#         ce, cd = detectar(carregar(pe)), detectar(carregar(pd))
#         if ce is not None and cd is not None:
#             o_s.append(OBJP); e_s.append(ce); d_s.append(cd)
