import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.camera import calibrate_camera, collect_calibration_points
from calibration.config import CalibrationConfig, DatasetPaths
from calibration.image_io import list_images
from calibration.reporting import (
    compare_with_ground_truth,
    create_output_archive,
    print_output_contents,
    print_summary,
    save_calibration_results,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calibração de câmera TA02")
    parser.add_argument("--device", default="android")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def run_calibration(args):
    config = CalibrationConfig(
        board_corners=(9, 4),
        square_size_mm=90,
        device=args.device,
        colab=args.colab,
        use_synthetic=args.synthetic,
        alpha=args.alpha,
    )
    paths = DatasetPaths(
        config.root_dir,
        config.use_synthetic,
        config.device,
    )

    # arquivos = listar(PASTA_CALIB)
    # testes = listar(PASTA_TESTE)
    # print(len(arquivos), "fotos de calibração |", len(testes), "de teste")
    calibration_images = list_images(paths.calibration)
    test_images = list_images(paths.test)
    print(
        len(calibration_images),
        "fotos de calibração |",
        len(test_images),
        "de teste",
    )

    if not calibration_images:
        raise RuntimeError(
            f"Nenhuma imagem de calibração encontrada em {paths.calibration}"
        )

    object_points, image_points, used_images, image_size = (
        collect_calibration_points(
            calibration_images,
            config.board_corners,
            config.square_size_mm,
        )
    )
    if not object_points or image_size is None:
        raise RuntimeError("Nenhum tabuleiro foi detectado nas imagens de calibração")

    result = calibrate_camera(object_points, image_points, image_size)
    resultados = {
        "n_fotos": len(calibration_images),
        "n_usadas": len(used_images),
        "rms": result.rms,
        "K": result.camera_matrix.tolist(),
        "dist": result.distortion.tolist(),
        "fov": [result.fov_x, result.fov_y],
        "erro_medio": float(result.reprojection_errors.mean()),
        "erro_max": float(result.reprojection_errors.max()),
        "tam": list(result.image_size),
    }

    compare_with_ground_truth(
        config,
        paths,
        result.camera_matrix,
        result.distortion,
        resultados,
    )

    # std_int = std_int.ravel()
    # A calibração atual do módulo camera.py não expõe std_int.
    std_int = np.zeros(9)
    lines = save_calibration_results(
        result,
        std_int,
        resultados,
        paths.metrics,
    )
    print_summary(result, resultados, lines)
    print_output_contents(paths.output)

    if args.zip:
        create_output_archive(paths.base, paths.output, config.colab)

    return result


def main():
    args = parse_arguments()
    run_calibration(args)


if __name__ == "__main__":
    main()