# 相机内参标定（棋盘格采集 → 标定 → 去畸变）

本目录用于 USB / 笔记本摄像头的针孔内参标定：采集棋盘格 → 解算 `K` 与畸变 `D` → 可选去畸变并得到去畸变图专用的 `K_new`。

## 环境

```powershell
conda activate testfree
# 若尚未安装：
# pip install opencv-python numpy
```

建议 PowerShell 先执行 `chcp 65001`，避免 `--help` 中文乱码。

## 数据流摘要

```text
摄像头实时帧
  → capture_chessboard.py     保存 calib_*.png + manifest.json
  → calibrate_intrinsics.py   角点检测 → calibrateCamera → K, D, RMS
  → undistort_images.py       （可选）remap 去畸变 → undist_*.png + K_new（D=0）
```

当前示例相机目录：`integrated_webcam/`（1280×720，内角点 11×8，方格 25 mm）。

| 文件 | 含义 |
|------|------|
| `calib_*.png` | 带畸变的原图（标定输入） |
| `manifest.json` | 采集元数据（棋盘尺寸等） |
| `camera_intrinsics.json` / `.yaml` | 原图配对：`K` + `D` |
| `undistorted/undist_*.png` | 去畸变图 |
| `undistorted/camera_intrinsics_undistorted.json` | 去畸变图配对：`K_new`，`D=0` |

## 快速开始

### 1. 采集棋盘格

```powershell
cd D:\calib_capture
python capture_chessboard.py --camera 0 --board 11 8 --width 1280 --height 720
```

- 画面显示 `BOARD OK` 时按 **空格 / 回车** 保存；`d` 删上一张；`n` 换姿态提示；`q` 退出。
- 尽量覆盖：正面 / 倾斜 / 远近 / 四角，建议 ≥15～20 张有效图。

### 2. 标定内参

```powershell
python calibrate_intrinsics.py --input-root D:/calib_capture/integrated_webcam --square-size 25 --workers 8
```

产出：`camera_intrinsics.json`、`camera_intrinsics.yaml`。关注 **RMS**（本机示例约 0.28 px，一般 <0.5 可用）。

### 3. 去畸变（可选）

```powershell
python undistort_images.py --input-root D:/calib_capture/integrated_webcam --alpha 0.0 --compare
```

- `--alpha 0`：裁掉无效像素，视野略缩、少黑边。
- `--alpha 1`：保留全部像素，边缘可能有黑边；`K_new` 会明显变化。
- `--compare`：输出左右对比图，便于肉眼检查直线是否被拉直。

## CAD + RGB：深度 / 位姿该用哪个 K？

**结论：未去畸变的 RGB 必须用原始 `K`，并且要带上 `D`；不能用 `K_new`。`K_new` 只属于已经 `remap` 过的去畸变图。**

| 你手里的 RGB | 该用的内参 | 说明 |
|--------------|------------|------|
| 摄像头原图（未去畸变） | **原始 `K` + `D`** | 像素落在畸变成像模型上；只用 `K` 忽略 `D` 会把边缘射线算歪 |
| 已跑过 `undistort_images.py` 的图 | **`K_new`，且 `D=0`** | 图与内参必须成对；禁止拿原图 `K` 配去畸变图 |

对「CAD mesh + RGB → 物体位姿 / 深度」的常见做法：

1. **多数开源 6D / 渲染对齐流水线（按理想针孔假设）**  
   先对 RGB 去畸变 → 用 **`K_new`** 做投影 / 反投影 / 渲染。不要把畸变原图塞进只认针孔的算法里却只填 `K`。

2. **坚持用畸变原图、不做 remap**  
   仍用 **原始 `K`**，但凡「像素 ↔ 射线 / 重投影」都要走带畸变的模型（例如先 `cv2.undistortPoints` 再反投影，或渲染时按 `D` 加畸变）。  
   **不能**把 `K_new` 套在原图上——`K_new` 是去畸变后的新针孔参数，和原图像素不对齐。

3. **「估计深度」若指 `Z = f * X / (u - cx)` 这类针孔公式**  
   - 原图：公式本身不完整，至少应对 `(u,v)` 先按 `K,D` 去畸变到理想像平面，再用原始 `fx,fy,cx,cy`；或整图去畸变后改用 `K_new`。  
   - 去畸变图：直接用 `K_new`。

一句话记忆：**图是哪一种，就用那一种的内参文件；原图 ↔ `camera_intrinsics.json`；去畸变图 ↔ `camera_intrinsics_undistorted.json`。**

## 与仿真（如 AirSim）对齐时

AirSim 等仿真相机通常是理想针孔（无畸变）。真机要对齐时：真机图先去畸变 → 用 `K_new` 推等效 FOV → 在仿真里设相同分辨率与 FOV，再谈伪真值 / 位姿迁移。

## 脚本说明

| 脚本 | 作用 |
|------|------|
| `capture_chessboard.py` | 交互采集棋盘格 |
| `calibrate_intrinsics.py` | 多进程角点 + `calibrateCamera` |
| `undistort_images.py` | 批量去畸变，写出 `K_new` |
| `_key_probe.py` | 调试 OpenCV `waitKey` 键码（可选） |

### 常用参数

```powershell
# 采集：指定相机索引与输出目录
python capture_chessboard.py --camera 0 --output-dir D:/calib_capture/integrated_webcam

# 标定：方格边长（毫米）必须与真实棋盘一致
python calibrate_intrinsics.py --square-size 25 --fix-aspect-ratio

# 去畸变：裁剪有效区并调整主点
python undistort_images.py --alpha 0.0 --crop --workers 8
```

## 标定质量自检

1. RMS 宜小（本项目笔记本摄像头约 0.2～0.5 px 较正常）。  
2. 打开 `undistorted/compare_*.png`：桌沿、门框应变直。  
3. `cx, cy` 应接近图像中心（1280×720 时约 640, 360；本机略偏属常见现象）。  
4. 姿态覆盖不足时，`cx/cy` 与高阶畸变项不稳定——补拍四角与倾斜姿态后重标。

## 依赖与复现

- Python 3 + OpenCV（`opencv-python`）+ NumPy  
- Windows 下多进程脚本已含 `if __name__ == "__main__":` / `freeze_support()`  
- 复现本机结果：同一批 `calib_*.png` + 相同 `--square-size` / `--board` 即可
