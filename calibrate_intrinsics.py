#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：用已采集的棋盘格图像求解相机内参矩阵 K 与畸变系数 D（张正友标定法），
      并输出逐张重投影误差、等效视场角 FOV，便于判断标定质量与迁移到仿真器。
      本脚本是 capture_chessboard.py 的下游：采图 → 本脚本 → 内参文件。

输入：--input-root 目录，包含 capture_chessboard.py 产出的
      calib_*.png 与 manifest.json（从 manifest 自动读取棋盘内角点数，可用 --board 覆盖）。

输出：写入 --input-root 同目录：
      camera_intrinsics.json   K、D、分辨率、RMS、逐张误差、等效 FOV
      camera_intrinsics.yaml   OpenCV FileStorage 格式，便于 C++/ROS 直接读取
      （加 --save-debug 时另存 debug_corners/ 角点可视化图）

依赖：pip install opencv-python

并行设计：角点检测是 CPU 密集且逐图独立，用 multiprocessing 并行。
      并行单位 = 单张图像；主进程负责建索引、汇总角点、调用 calibrateCamera（该步不可并行）；
      worker 只读图不写盘，无写冲突；--workers 1 与多进程共用同一套逻辑，便于调试对照。

运行示例：
      python calibrate_intrinsics.py

      python calibrate_intrinsics.py --input-root D:/calib_capture/integrated_webcam ^
          --square-size 25.0 --workers 8 --save-debug
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_INPUT_ROOT = "D:/calib_capture/integrated_webcam"


def parse_board_size(text):
    """把 '11x8' 解析为 (cols, rows) 内角点数。"""
    parts = text.lower().replace("*", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("棋盘规格格式应为 列x行，例如 11x8")
    return int(parts[0]), int(parts[1])


def parse_args():
    parser = argparse.ArgumentParser(
        description="棋盘格相机内参标定：读取已采集图像，求解内参矩阵 K 与畸变系数 D，并输出重投影误差与等效视场角。",
        epilog="示例：python calibrate_intrinsics.py --input-root D:/calib_capture/integrated_webcam --square-size 25.0",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT,
                        help="标定图所在目录，决定读取哪批 calib_*.png（默认 %s）" % DEFAULT_INPUT_ROOT)
    parser.add_argument("--board", type=parse_board_size, default=None,
                        help="棋盘内角点数，格式 列x行。不填则从 manifest.json 自动读取")
    parser.add_argument("--square-size", type=float, default=25.0,
                        help="棋盘单格实际边长（毫米）。只影响外参平移量的单位，不影响 K 与畸变（默认 25.0）")
    parser.add_argument("--workers", type=int, default=None,
                        help="角点检测的并行进程数，影响耗时不影响结果（默认 CPU 核数-1）")
    parser.add_argument("--fix-aspect-ratio", action="store_true",
                        help="布尔开关，默认 False。开启后强制 fx=fy，适合像素为正方形的普通相机")
    parser.add_argument("--error-threshold", type=float, default=1.0,
                        help="逐张重投影误差告警阈值（像素），超过的图会被标记为可疑（默认 1.0）")
    parser.add_argument("--save-debug", action="store_true",
                        help="布尔开关，默认 False。开启后把角点可视化图写入 debug_corners/ 便于排查")
    return parser.parse_args()


def load_board_from_manifest(input_dir):
    """从 manifest.json 读取内角点数，读不到返回 None。"""
    manifest = input_dir / "manifest.json"
    if not manifest.exists():
        return None
    with open(manifest, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    corners = data.get("board_inner_corners")
    return (int(corners[0]), int(corners[1])) if corners else None


def detect_corners_task(job):
    """worker：单张图检测棋盘角点。返回 (路径, 是否成功, 角点, 图像尺寸)。

    WHY: 该函数必须是模块级顶层函数，否则 Windows 的 spawn 方式无法 pickle 传给子进程。
    """
    path_str, board_size = job
    image = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if image is None:
        return path_str, False, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size_wh = (gray.shape[1], gray.shape[0])
    # WHY: SB 版本对模糊、透视畸变更鲁棒，且自带亚像素精度；失败时回退经典算法。
    found, corners = cv2.findChessboardCornersSB(
        gray, board_size, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    if not found:
        return path_str, False, None, size_wh
    return path_str, True, np.asarray(corners, dtype=np.float32), size_wh


def detect_all_corners(image_paths, board_size, workers):
    """主进程：分派检测任务并汇总结果。返回 (成功列表, 失败列表, 图像尺寸)。"""
    jobs = [(str(p), board_size) for p in image_paths]
    if workers <= 1:
        results = [detect_corners_task(job) for job in jobs]
    else:
        # WHY: Windows 下显式用 spawn 上下文，避免继承父进程状态导致的不确定行为。
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(detect_corners_task, jobs)

    ok, failed, image_size = [], [], None
    for path_str, found, corners, size_wh in results:
        if size_wh is not None:
            image_size = image_size or size_wh
            if size_wh != image_size:
                raise RuntimeError("图像分辨率不一致: %s 是 %s，期望 %s" % (path_str, size_wh, image_size))
        (ok if found else failed).append((path_str, corners))
    return ok, failed, image_size


def build_object_points(board_size, square_size):
    """生成棋盘在自身坐标系下的 3D 点（Z=0 平面），单位与 square_size 一致。"""
    cols, rows = board_size
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return grid * square_size


def compute_per_view_errors(obj_points, img_points, rvecs, tvecs, K, D):
    """逐张计算重投影 RMS 误差（像素）。"""
    errors = []
    for i, objp in enumerate(obj_points):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, D)
        diff = img_points[i].reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return errors


def compute_fov(K, image_size):
    """由 K 反推等效水平/垂直视场角（度），用于对齐仿真器的 FOV_Degrees 设置。"""
    width, height = image_size
    fov_x = 2.0 * math.degrees(math.atan(width / (2.0 * K[0, 0])))
    fov_y = 2.0 * math.degrees(math.atan(height / (2.0 * K[1, 1])))
    return fov_x, fov_y


def save_debug_images(ok_results, board_size, output_dir):
    """把检出的角点画回图上，便于人工确认检测是否正确。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for path_str, corners in ok_results:
        image = cv2.imread(path_str, cv2.IMREAD_COLOR)
        if image is None:
            continue
        cv2.drawChessboardCorners(image, board_size, corners, True)
        cv2.imwrite(str(output_dir / Path(path_str).name), image)


def write_results(input_dir, payload, K, D, image_size):
    """写出 JSON 与 OpenCV YAML 两份内参文件。"""
    with open(input_dir / "camera_intrinsics.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    fs = cv2.FileStorage(str(input_dir / "camera_intrinsics.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", D)
    fs.release()


def report(payload, errors, files, threshold):
    """打印标定结果与可疑图像。"""
    K = payload["camera_matrix"]
    print("\n=== 标定结果 ===")
    print("分辨率      : %dx%d" % tuple(payload["image_size"]))
    print("有效图像    : %d 张" % payload["used_images"])
    print("整体 RMS    : %.4f 像素" % payload["rms"])
    print("fx, fy      : %.3f, %.3f" % (K[0][0], K[1][1]))
    print("cx, cy      : %.3f, %.3f" % (K[0][2], K[1][2]))
    print("畸变 D      : %s" % np.round(payload["distortion"], 6).tolist())
    print("等效 FOV    : 水平 %.2f 度, 垂直 %.2f 度" % tuple(payload["fov_degrees"]))
    suspects = [(f, e) for f, e in zip(files, errors) if e > threshold]
    if suspects:
        print("\n重投影误差偏大（> %.2f 像素），建议重拍或剔除：" % threshold)
        for name, err in suspects:
            print("  %-20s %.4f" % (name, err))
    else:
        print("\n所有图像重投影误差均低于 %.2f 像素" % threshold)


def main():
    args = parse_args()
    input_dir = Path(args.input_root)
    if not input_dir.is_dir():
        raise RuntimeError("目录不存在: %s" % input_dir)

    board_size = args.board or load_board_from_manifest(input_dir)
    if board_size is None:
        raise RuntimeError("未指定 --board 且 manifest.json 中没有 board_inner_corners")

    image_paths = sorted(input_dir.glob("calib_*.png"))
    if len(image_paths) < 5:
        raise RuntimeError("图像太少（%d 张），标定至少需要 5 张，建议 20 张以上" % len(image_paths))

    workers = args.workers or max(1, (mp.cpu_count() or 4) - 1)
    print("输入目录: %s" % input_dir)
    print("图像数量: %d, 棋盘内角点: %dx%d, 并行进程: %d"
          % (len(image_paths), board_size[0], board_size[1], workers))

    ok_results, failed, image_size = detect_all_corners(image_paths, board_size, workers)
    print("角点检出: %d 成功 / %d 失败" % (len(ok_results), len(failed)))
    for path_str, _ in failed:
        print("  检测失败(已跳过): %s" % Path(path_str).name)
    if len(ok_results) < 5:
        raise RuntimeError("成功检出角点的图像不足 5 张，请检查 --board 是否与实际棋盘一致")

    objp = build_object_points(board_size, args.square_size)
    obj_points = [objp for _ in ok_results]
    img_points = [corners for _, corners in ok_results]

    flags = cv2.CALIB_FIX_ASPECT_RATIO if args.fix_aspect_ratio else 0
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags)

    errors = compute_per_view_errors(obj_points, img_points, rvecs, tvecs, K, D)
    fov_x, fov_y = compute_fov(K, image_size)
    files = [Path(p).name for p, _ in ok_results]

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(input_dir),
        "board_inner_corners": list(board_size),
        "square_size_mm": args.square_size,
        "image_size": list(image_size),
        "used_images": len(ok_results),
        "failed_images": [Path(p).name for p, _ in failed],
        "rms": float(rms),
        "camera_matrix": K.tolist(),
        "distortion": D.ravel().tolist(),
        "fov_degrees": [fov_x, fov_y],
        "per_view_errors": dict(zip(files, [round(e, 5) for e in errors])),
    }

    write_results(input_dir, payload, K, D, image_size)
    if args.save_debug:
        save_debug_images(ok_results, board_size, input_dir / "debug_corners")
        print("角点可视化已写入: %s" % (input_dir / "debug_corners"))

    report(payload, errors, files, args.error_threshold)
    print("\n已写入: %s" % (input_dir / "camera_intrinsics.json"))
    print("已写入: %s" % (input_dir / "camera_intrinsics.yaml"))
    return 0


if __name__ == "__main__":
    # WHY: Windows spawn 方式下必须有 main 保护，否则子进程会重复执行模块顶层代码。
    mp.freeze_support()
    sys.exit(main())
