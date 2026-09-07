#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：USB/内置摄像头棋盘格标定图采集器。实时预览 + 角点检测，按键保存多角度棋盘格照片，
      为后续相机内参标定（求解 fx/fy/cx/cy 与畸变系数）准备原始数据。
      本脚本只负责“采图”，不计算内参。

输入：摄像头设备（Windows 下通过 cv2.CAP_DSHOW 按索引打开，内置 Webcam 一般是 0）。
      物理棋盘格标定板，参数用内角点数描述（如 9x6 表示 9 列 x 6 行内角点，即 10x7 个格子）。

输出：--output-root 目录下：
      calib_000001.png, calib_000002.png, ...   原始 BGR 帧（未压缩绘制，不含角点叠加）
      manifest.json                             每帧记录：时间戳、分辨率、是否检出角点、姿态标签

依赖：pip install opencv-python
      （numpy 随 opencv-python 自动安装）

为何本脚本未使用多进程：任务是单摄像头交互式采集，瓶颈在相机出帧与人工摆位，
      不是 CPU 密集批处理；多进程无法共享同一个 VideoCapture 且会破坏按键交互。

操作按键：
      空格 / Enter : 保存当前帧
      d            : 删除最近一张（同时回退 manifest）
      n            : 手动切到下一个姿态提示
      q / Esc      : 退出

运行示例：
      python capture_chessboard.py

      python capture_chessboard.py --camera-index 0 --board 9x6 --width 1280 --height 720 ^
          --output-root D:/calib_capture/integrated_webcam --target-per-pose 3
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# WHY: cv2.putText 不支持中文，画面叠加统一用 ASCII，中文说明只在终端打印，避免出现方块乱码。
POSE_HINTS = [
    ("front", "正视：棋盘正对镜头，约占画面 60%"),
    ("tilt-left", "左倾：绕竖直轴向左转 30~45 度"),
    ("tilt-right", "右倾：绕竖直轴向右转 30~45 度"),
    ("tilt-up", "上仰：棋盘上沿后仰 30~45 度"),
    ("tilt-down", "下俯：棋盘下沿后仰 30~45 度"),
    ("near", "拉近：棋盘几乎充满画面"),
    ("far", "拉远：棋盘约占画面 30%"),
    ("corner", "移到画面四角，各拍一张"),
]

DEFAULT_OUTPUT_ROOT = "D:/calib_capture/integrated_webcam"


def parse_board_size(text):
    """把 '9x6' 解析为 (cols, rows) 内角点数。"""
    parts = text.lower().replace("*", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("棋盘规格格式应为 列x行，例如 9x6")
    cols, rows = int(parts[0]), int(parts[1])
    if cols < 3 or rows < 3:
        raise argparse.ArgumentTypeError("内角点数至少 3x3")
    return cols, rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="棋盘格标定图采集器：实时预览摄像头画面，按空格保存多角度棋盘格照片，供后续内参标定使用。",
        epilog="示例：python capture_chessboard.py --camera-index 0 --board 9x6 --width 1280 --height 720",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--camera-index", type=int, default=0,
                        help="摄像头索引，决定打开哪个设备。Windows 内置 Webcam 通常为 0（默认 0）")
    parser.add_argument("--board", type=parse_board_size, default="9x6",
                        help="棋盘格内角点数，格式 列x行。注意是黑白格交点数，不是格子数（默认 9x6）")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT,
                        help="图片与 manifest.json 的输出目录，影响写盘路径（默认 %s）" % DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=1280,
                        help="请求的采集宽度（像素）。相机不支持时会回落到实际分辨率（默认 1280）")
    parser.add_argument("--height", type=int, default=720,
                        help="请求的采集高度（像素）。相机不支持时会回落到实际分辨率（默认 720）")
    parser.add_argument("--target-per-pose", type=int, default=3,
                        help="每个姿态建议采集张数，仅用于自动切换提示，不限制实际保存（默认 3）")
    parser.add_argument("--detect-every", type=int, default=2,
                        help="每隔几帧做一次角点检测，越大预览越流畅但提示越滞后（默认 2）")
    parser.add_argument("--allow-no-board", action="store_true",
                        help="布尔开关，默认 False。开启后未检出角点也允许保存；默认必须检出角点才能保存，避免废图")
    return parser.parse_args()


def open_camera(index, width, height):
    """打开摄像头并尽量设置分辨率，返回 (cap, 实际宽, 实际高)。"""
    # WHY: Windows 上 CAP_DSHOW 比默认 MSMF 后端启动快且更少卡死。
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头索引 %d，请确认设备未被其他程序占用" % index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, actual_w, actual_h


def detect_chessboard(frame_bgr, board_size):
    """检测棋盘格内角点，返回 (found, corners)。corners 为亚像素精修后的坐标。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # WHY: FAST_CHECK 让无棋盘的帧快速返回，保证预览帧率。
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def put_text(canvas, text, org, color, scale=0.6):
    """带黑色描边的文字，保证在明暗背景下都看得清。"""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(canvas, found, corners, board_size, saved, pose_label, pose_done, target,
                 toast="", toast_ok=True, last_key=-1):
    """在预览副本上叠加角点与状态文字（纯 ASCII）。"""
    if found and corners is not None:
        cv2.drawChessboardCorners(canvas, board_size, corners, True)
    color = (0, 220, 0) if found else (0, 0, 255)
    status = "BOARD OK" if found else "NO BOARD"
    lines = [
        "%s  saved=%d" % (status, saved),
        "pose: %s (%d/%d)" % (pose_label, pose_done, target),
        "SPACE=save  d=delete  n=next pose  q=quit",
        "last key=%d" % last_key,
    ]
    for i, text in enumerate(lines):
        put_text(canvas, text, (12, 28 + i * 26), color)
    # WHY: 保存被拒/成功必须在画面上可见，否则用户盯着窗口时以为按键没生效。
    if toast:
        h = canvas.shape[0]
        put_text(canvas, toast, (12, h - 24), (0, 220, 0) if toast_ok else (0, 0, 255), 0.8)
    return canvas


def save_frame(raw_frame, output_dir, index, found, pose_label):
    """保存原始帧（不带叠加），返回 manifest 记录字典。"""
    name = "calib_%06d.png" % index
    path = output_dir / name
    # WHY: 写盘用未叠加的原始帧，叠加只用于预览；否则角点绘制会污染标定输入。
    if not cv2.imwrite(str(path), raw_frame):
        raise RuntimeError("写入失败: %s" % path)
    h, w = raw_frame.shape[:2]
    return {
        "file": name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "width": w,
        "height": h,
        "board_found": bool(found),
        "pose_hint": pose_label,
    }


def write_manifest(output_dir, meta, records):
    """把采集元信息与逐帧记录写为 manifest.json。"""
    payload = dict(meta)
    payload["frame_count"] = len(records)
    payload["frames"] = records
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def delete_last(output_dir, records):
    """删除最近一张图片并回退 manifest 记录，返回提示文字。"""
    if not records:
        return "没有可删除的图片"
    record = records.pop()
    target = output_dir / record["file"]
    if target.exists():
        target.unlink()
        return "已删除 %s" % record["file"]
    return "文件不存在，仅回退记录: %s" % record["file"]


def print_pose_guide():
    print("\n姿态采集建议（每个姿态多拍几张，总计 20~30 张）：")
    for label, desc in POSE_HINTS:
        print("  %-11s %s" % (label, desc))
    print()


def main():
    args = parse_args()
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, width, height = open_camera(args.camera_index, args.width, args.height)
    print("摄像头 %d 已打开，实际分辨率 %dx%d" % (args.camera_index, width, height))
    print("棋盘内角点: %dx%d, 输出目录: %s" % (args.board[0], args.board[1], output_dir))
    print("保存策略: %s" % ("允许未检出角点保存" if args.allow_no_board else "必须检出角点才能保存"))
    print_pose_guide()

    meta = {
        "camera_index": args.camera_index,
        "board_inner_corners": list(args.board),
        "requested_resolution": [args.width, args.height],
        "actual_resolution": [width, height],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    records = []
    pose_idx, pose_done, frame_no = 0, 0, 0
    found, corners = False, None
    toast, toast_ok, toast_ttl, last_key = "", True, 0, -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读取帧失败，跳过")
                continue
            frame_no += 1
            # WHY: 隔帧检测降低 CPU 占用，保证预览不卡；保存时会对当前帧重新检测一次。
            if frame_no % max(1, args.detect_every) == 0:
                found, corners = detect_chessboard(frame, args.board)

            pose_label = POSE_HINTS[pose_idx][0]
            toast_ttl = max(0, toast_ttl - 1)
            canvas = draw_overlay(frame.copy(), found, corners, args.board,
                                  len(records), pose_label, pose_done, args.target_per_pose,
                                  toast if toast_ttl else "", toast_ok, last_key)
            cv2.imshow("chessboard capture", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                last_key = key
            if key in (ord("q"), 27):
                break
            # WHY: 不同键盘/IME 下回车可能是 13 或 10，两个都收。
            if key in (ord(" "), 13, 10):
                hit, pts = detect_chessboard(frame, args.board)
                if not hit and not args.allow_no_board:
                    toast, toast_ok, toast_ttl = "REJECTED: no chessboard detected", False, 45
                    print("未检出角点，本帧未保存（确认 --board 内角点数，或加 --allow-no-board 放宽）")
                    continue
                records.append(save_frame(frame, output_dir, len(records) + 1, hit, pose_label))
                write_manifest(output_dir, meta, records)
                pose_done += 1
                toast, toast_ok, toast_ttl = "SAVED %s" % records[-1]["file"], True, 45
                print("已保存 %s (角点=%s, 姿态=%s, 累计 %d 张)"
                      % (records[-1]["file"], hit, pose_label, len(records)))
                if pose_done >= args.target_per_pose and pose_idx < len(POSE_HINTS) - 1:
                    pose_idx, pose_done = pose_idx + 1, 0
                    print(">>> 切换姿态: %s — %s" % POSE_HINTS[pose_idx])
            elif key == ord("d"):
                message = delete_last(output_dir, records)
                write_manifest(output_dir, meta, records)
                pose_done = max(0, pose_done - 1)
                toast, toast_ok, toast_ttl = "DELETED", True, 45
                print(message)
            elif key == ord("n"):
                pose_idx, pose_done = (pose_idx + 1) % len(POSE_HINTS), 0
                toast, toast_ok, toast_ttl = "pose -> %s" % POSE_HINTS[pose_idx][0], True, 45
                print(">>> 切换姿态: %s — %s" % POSE_HINTS[pose_idx])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        write_manifest(output_dir, meta, records)
        print("\n采集结束，共 %d 张，已写入 %s" % (len(records), output_dir / "manifest.json"))
        if len(records) < 20:
            print("提示：标定建议至少 20 张，且覆盖不同角度与远近。")


if __name__ == "__main__":
    sys.exit(main())
