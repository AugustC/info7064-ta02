import json
import os
import shutil

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration.camera import CalibrationResults
from calibration.config import CalibrationConfig, DatasetPaths


def save_figure(name, figures_directory):
    # def salvar_fig(nome):
    #     plt.savefig(os.path.join(PASTA_FIGURAS, nome), dpi=150, bbox_inches="tight")
    plt.savefig(
        os.path.join(figures_directory, name),
        dpi=150,
        bbox_inches="tight",
    )


def compare_with_ground_truth(
    config: CalibrationConfig,
    paths: DatasetPaths,
    K,
    dist,
    results,
):
    # if USAR_SINTETICAS:
    #     verd = np.load(os.path.join(ENTRADA, "verdade_sintetica.npz"), allow_pickle=True)
    #     nomes = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    #     real = [verd["K"][0,0], verd["K"][1,1], verd["K"][0,2], verd["K"][1,2], *verd["dist"]]
    #     est  = [K[0,0], K[1,1], K[0,2], K[1,2], *dist]
    #     print(f"{'':4}{'real':>11}{'estimado':>11}{'erro':>10}")
    #     tabela = []
    #     for n, r, e in zip(nomes, real, est):
    #         print(f"{n:4}{r:11.4f}{e:11.4f}{e - r:10.4f}")
    #         tabela.append([n, float(r), float(e)])
    #     RESULTADOS["verdade"] = tabela
    if not config.use_synthetic:
        return
    ground_truth = np.load(
        os.path.join(paths.input, "verdade_sintetica.npz"),
        allow_pickle=True,
    )
    names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    real = [
        ground_truth["K"][0, 0],
        ground_truth["K"][1, 1],
        ground_truth["K"][0, 2],
        ground_truth["K"][1, 2],
        *ground_truth["dist"],
    ]
    estimated = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist]
    print(f"{'':4}{'real':>11}{'estimado':>11}{'erro':>10}")
    table = []
    for name, real_value, estimated_value in zip(names, real, estimated):
        print(
            f"{name:4}{real_value:11.4f}{estimated_value:11.4f}"
            f"{estimated_value - real_value:10.4f}"
        )
        table.append([name, float(real_value), float(estimated_value)])
    results["verdade"] = table


def save_calibration_results(
    result: CalibrationResults,
    std_int,
    resultados,
    metrics_directory,
):
    # with open(os.path.join(PASTA_DADOS, "resultados.json"), "w") as f:
    #     json.dump(RESULTADOS, f, indent=2, ensure_ascii=False)
    #
    # np.savetxt(os.path.join(PASTA_DADOS, "matriz_K.txt"), K, fmt="%.6f")
    # np.savetxt(os.path.join(PASTA_DADOS, "distorcao.txt"), dist.reshape(1, -1), fmt="%.8f",
    #            header="k1 k2 p1 p2 k3")
    #
    # br = lambda x, n=2: f"{x:.{n}f}".replace(".", "{,}")
    # linhas = []
    # for nome, val, sd in [("f_x", K[0,0], std_int[0]), ("f_y", K[1,1], std_int[1]),
    #                       ("c_x", K[0,2], std_int[2]), ("c_y", K[1,2], std_int[3])]:
    #     linhas.append(f"${nome}$ (px) & {br(val)} & {br(sd)} \\\\")
    # for nome, val, sd in zip(["k_1","k_2","p_1","p_2","k_3"], dist, std_int[4:9]):
    #     linhas.append(f"${nome}$ & {br(val,4)} & {br(sd,4)} \\\\")
    # linhas.append(f"RMS (px) & \\multicolumn{{2}}{{c}}{{{br(rms,3)}}} \\\\")
    # linhas.append("")
    # linhas.append("% experimento 3D -> 2D")
    # for n, (nome, dcam, em, emax, e0) in enumerate(RESULTADOS.get("experimento", []), 1):
    #     linhas.append(f"Teste {n} & {dcam:.0f} & {br(em,3)} & {br(emax,3)} & {br(e0)} \\\\")
    # with open(os.path.join(PASTA_DADOS, "tabelas_latex.txt"), "w") as f:
    #     f.write("\n".join(linhas))
    metrics_directory = str(metrics_directory)
    os.makedirs(metrics_directory, exist_ok=True)
    with open(os.path.join(metrics_directory, "resultados.json"), "w") as file:
        json.dump(resultados, file, indent=2, ensure_ascii=False)

    np.savetxt(
        os.path.join(metrics_directory, "matriz_K.txt"),
        result.camera_matrix,
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(metrics_directory, "distorcao.txt"),
        result.distortion.reshape(1, -1),
        fmt="%.8f",
        header="k1 k2 p1 p2 k3",
    )

    def format_decimal(value, decimals=2):
        return f"{value:.{decimals}f}".replace(".", "{,}")

    lines = []
    for name, value, standard_deviation in [
        ("f_x", result.camera_matrix[0, 0], std_int[0]),
        ("f_y", result.camera_matrix[1, 1], std_int[1]),
        ("c_x", result.camera_matrix[0, 2], std_int[2]),
        ("c_y", result.camera_matrix[1, 2], std_int[3]),
    ]:
        lines.append(
            f"${name}$ (px) & {format_decimal(value)} & "
            f"{format_decimal(standard_deviation)} \\\\"
        )
    for name, value, standard_deviation in zip(
        ["k_1", "k_2", "p_1", "p_2", "k_3"],
        result.distortion,
        std_int[4:9],
    ):
        lines.append(
            f"${name}$ & {format_decimal(value, 4)} & "
            f"{format_decimal(standard_deviation, 4)} \\\\"
        )
    lines.append(
        f"RMS (px) & \\multicolumn{{2}}{{c}}{{"
        f"{format_decimal(result.rms, 3)}}} \\\\"
    )
    lines.append("")
    lines.append("% experimento 3D -> 2D")
    for number, (name, distance, mean_error, max_error, no_distortion_error) in enumerate(
        resultados.get("experimento", []),
        1,
    ):
        lines.append(
            f"Teste {number} & {distance:.0f} & "
            f"{format_decimal(mean_error, 3)} & "
            f"{format_decimal(max_error, 3)} & "
            f"{format_decimal(no_distortion_error)} \\\\"
        )
    with open(os.path.join(metrics_directory, "tabelas_latex.txt"), "w") as file:
        file.write("\n".join(lines))
    return lines


def print_summary(result, resultados, lines):
    # print("===== RESUMO =====")
    # print(f"fotos: {RESULTADOS['n_usadas']}/{RESULTADOS['n_fotos']} | resolução {TAM[0]}x{TAM[1]}")
    # print(f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    # print("dist =", np.round(dist, 5))
    # print(f"RMS = {rms:.4f} px | FOV {fov_x:.1f}° x {fov_y:.1f}°")
    # print("\n--- tabelas LaTeX (também em dados/tabelas_latex.txt) ---")
    # print("\n".join(linhas))
    print("===== RESUMO =====")
    print(
        f"fotos: {resultados['n_usadas']}/{resultados['n_fotos']} | "
        f"resolução {result.image_size[0]}x{result.image_size[1]}"
    )
    K = result.camera_matrix
    print(
        f"fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} "
        f"cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}"
    )
    print("dist =", np.round(result.distortion, 5))
    print(
        f"RMS = {result.rms:.4f} px | "
        f"FOV {result.fov_x:.1f}° x {result.fov_y:.1f}°"
    )
    print("\n--- tabelas LaTeX (também em dados/tabelas_latex.txt) ---")
    print("\n".join(lines))


def print_output_contents(output_directory):
    # print("\n===== CONTEÚDO DA PASTA DE SAÍDA =====")
    # for raiz, _, arqs in sorted(os.walk(SAIDA)):
    #     nivel = raiz.replace(SAIDA, "").count(os.sep)
    #     print("   " * nivel + os.path.basename(raiz) + "/", f"({len(arqs)} arquivos)")
    print("\n===== CONTEÚDO DA PASTA DE SAÍDA =====")
    for root, _, files in sorted(os.walk(output_directory)):
        level = root.replace(str(output_directory), "").count(os.sep)
        print("   " * level + os.path.basename(root) + "/", f"({len(files)} arquivos)")


def create_output_archive(base_directory, output_directory, colab=False):
    # zip_saida = shutil.make_archive(os.path.join(BASE, "saida_calibracao"), "zip", SAIDA)
    # print("zip criado:", zip_saida)
    # if NO_COLAB:
    #     from google.colab import files
    #     files.download(zip_saida)
    archive = shutil.make_archive(
        os.path.join(base_directory, "saida_calibracao"),
        "zip",
        output_directory,
    )
    print("zip criado:", archive)
    if colab:
        from google.colab import files
        files.download(archive)
    return archive
