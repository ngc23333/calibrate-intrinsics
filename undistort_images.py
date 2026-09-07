#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：用标定得到的 K 与畸变系数 D 对图像做去畸变（矫正桶形/枕形畸变），
      并输出去畸变后对应的**新内参 K_new**。
      注意：去畸变是标定的下游操作，不是标定的前置步骤。

输入：--intrinsics 指向 calibrate_intrinsics.py 产出的 camera_intrinsics.json
      --input-root 待去畸变的图像目录（默认与内参同目录，处理 calib_*.png）

输出：--output-dir（默认 <input-root>/undistorted）
      undist_*.png                       去畸变后的图像
      camera_intrinsics_undistorted.json 去畸变图对应的新内参 K_new（畸变为全 0）
      （加 --compare 时另存 compare_*.png 原图/矫正图左右对比）

依赖：pip install opencv-python

关于 --alpha：
      alpha=0  裁掉所有无效像素，画面无黑边但视野变小，K_new 的 fx/cx 都会变
      alpha=1  保留全部原始像素，边缘出现黑色弯曲区域，视野最大
      中间值在两者之间插值。默认 0.0。

并行设计：remap 是逐图独立的 CPU 密集操作，用 multiprocessing 并行。
      并行单位 = 单张图像；主进程负责读内参、算 K_new 与重映射表、汇总统计；
      worker 用 initializer 一次性构建 map 后各自读图、remap、写盘到同一输出目录，
      文件名由输入名唯一决定，无写冲突；--workers 1 走串行分支，便于调试对照。

运行示例：
      python undistort_images.py

      python undistort_images.py --input-root D:/calib_capture/integrated_webcam ^
          --alpha 0.0 --workers 8 --compare
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ROOT = "D:/calib_capture/integrated_webcam"

# WHY: 重映射表体积大且只依赖 K/D/alpha，用进程级全局变量在 initializer 里构建一次，
#      避免每张图重复计算，也避免把大数组反复 pickle 传给子进程。
_MAP_X = None
_MAP_Y = None
_ROI = None
_COMPARE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="图像去畸变：读取标定内参，矫正镜头畸变，并输出去畸变图对应的新内参矩阵。",
        epilog="示例：python undistort_images.py --alpha 0.0 --compare",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input-root", type=str, default=DEFAULT_ROOT,
                        help="待处理图像目录，决定读取哪批 --pattern 匹配的图（默认 %s）" % DEFAULT_ROOT)
    parser.add_argument("--intrinsics", type=str, default=None,
                        help="内参 JSON 路径。不填则用 <input-root>/camera_intrinsics.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，影响写盘路径。不填则用 <input-root>/undistorted")
    parser.add_argument("--pattern", type=str, default="calib_*.png",
                        help="输入文件名通配符，决定处理哪些图（默认 calib_*.png）")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="自由缩放系数，影响视野与黑边：0=裁掉黑边视野变小，1=保留全部像素有黑边（默认 0.0）")
    parser.add_argument("--crop", action="store_true",
                        help="布尔开关，默认 False。开启后按有效区域 ROI 裁剪，输出分辨率会变小，K_new 主点随之平移")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数，影响耗时不影响结果（默认 CPU 核数-1）")
    parser.add_argument("--compare", action="store_true",
                        help="布尔开关，默认 False。开启后额外输出原图与矫正图的左右拼接对比图")
    return parser.parse_args()


def load_intrinsics(path):
    """读取内参 JSON，返回 (K, D, (width, height))。"""
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    D = np.array(data["distortion"], dtype=np.float64).reshape(1, -1)
    size = (int(data["image_size"][0]), int(data["image_size"][1]))
    return K, D, size


def init_worker(K, D, K_new, size, roi, compare):
    """worker 初始化：构建一次重映射表，后续每张图直接 remap。"""
    global _MAP_X, _MAP_Y, _ROI, _COMPARE
    _MAP_X, _MAP_Y = cv2.initUndistortRectifyMap(
        K, D, None, K_new, size, cv2.CV_32FC1)
    _ROI = roi
    _COMPARE = compare


def undistort_task(job):
    """worker：单张图去畸变并写盘。返回 (输入名, 是否成功, 输出尺寸)。"""
    src_path, dst_path, compare_path = job
    image = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if image is None:
        return Path(src_path).name, False, None

    fixed = cv2.remap(image, _MAP_X, _MAP_Y, cv2.INTER_LINEAR)
    if _ROI is not None:
        x, y, w, h = _ROI
        fixed = fixed[y:y + h, x:x + w]

    cv2.imwrite(dst_path, fixed)
    if _COMPARE and compare_path:
        # WHY: 对比图仅用于人眼检查，把矫正结果缩放回原高度再左右拼接。
        scaled = cv2.resize(fixed, (int(fixed.shape[1] * image.shape[0] / fixed.shape[0]),
                                    image.shape[0]))
        cv2.imwrite(compare_path, np.hstack([image, scaled]))
    return Path(src_path).name, True, (fixed.shape[1], fixed.shape[0])


def build_jobs(paths, output_dir, compare):
    """生成 (输入路径, 输出路径, 对比图路径) 任务列表。"""
    jobs = []
    for path in paths:
        dst = output_dir / ("undist_" + path.name)
        cmp_path = str(output_dir / ("compare_" + path.name)) if compare else None
        jobs.append((str(path), str(dst), cmp_path))
    return jobs


def run_jobs(jobs, workers, init_args):
    """按 workers 决定串行或多进程执行。"""
    if workers <= 1:
        init_worker(*init_args)
        return [undistort_task(job) for job in jobs]
    # WHY: Windows 下显式 spawn，并用 initializer 把重映射表放进子进程一次性构建。
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=init_worker, initargs=init_args) as pool:
        return pool.map(undistort_task, jobs)


def save_new_intrinsics(output_dir, K_new, out_size, alpha, cropped, source):
    """写出去畸变图对应的新内参（畸变已消除，故 D 全 0）。"""
    payload = {
        "note": "适用于去畸变后的图像；畸变已消除，D 为全 0。不要与原图混用。",
        "source_intrinsics": str(source),
        "alpha": alpha,
        "cropped_to_roi": bool(cropped),
        "image_size": [int(out_size[0]), int(out_size[1])],
        "camera_matrix": K_new.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    with open(output_dir / "camera_intrinsics_undistorted.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return payload


def main():
    args = parse_args()
    input_dir = Path(args.input_root)
    intr_path = Path(args.intrinsics) if args.intrinsics else input_dir / "camera_intrinsics.json"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "undistorted"

    if not intr_path.exists():
        raise RuntimeError("找不到内参文件: %s，请先运行 calibrate_intrinsics.py" % intr_path)
    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise RuntimeError("目录 %s 下没有匹配 %s 的图像" % (input_dir, args.pattern))
    output_dir.mkdir(parents=True, exist_ok=True)

    K, D, size = load_intrinsics(intr_path)
    # WHY: alpha 决定保留多少原始像素；K_new 是去畸变图的真实内参，与原 K 不同。
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, D, size, args.alpha, size)
    use_roi = roi if (args.crop and roi[2] > 0 and roi[3] > 0) else None

    workers = args.workers or max(1, (mp.cpu_count() or 4) - 1)
    print("内参文件: %s" % intr_path)
    print("图像数量: %d, 分辨率 %dx%d, alpha=%.2f, 并行进程 %d"
          % (len(paths), size[0], size[1], args.alpha, workers))
    print("原始 K : fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))
    print("新的 K : fx=%.2f fy=%.2f cx=%.2f cy=%.2f"
          % (K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]))
    print("有效区 ROI: %s%s" % (roi, "（已裁剪）" if use_roi else "（未裁剪）"))

    jobs = build_jobs(paths, output_dir, args.compare)
    init_args = (K, D, K_new, size, use_roi, args.compare)
    results = run_jobs(jobs, workers, init_args)

    ok = [r for r in results if r[1]]
    out_size = ok[0][2] if ok else size
    if use_roi:
        # WHY: 裁剪后主点要减去 ROI 左上角偏移，否则后续投影会整体错位。
        K_new = K_new.copy()
        K_new[0, 2] -= use_roi[0]
        K_new[1, 2] -= use_roi[1]

    save_new_intrinsics(output_dir, K_new, out_size, args.alpha, bool(use_roi), intr_path)

    print("\n完成: %d/%d 张, 输出尺寸 %dx%d" % (len(ok), len(results), out_size[0], out_size[1]))
    for name, success, _ in results:
        if not success:
            print("  处理失败: %s" % name)
    print("图像输出: %s" % output_dir)
    print("新内参  : %s" % (output_dir / "camera_intrinsics_undistorted.json"))
    print("\n提醒：后续对去畸变图做任何投影计算，必须用新内参，且 D 视为全 0。")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
