# DIC 超分辨率三维重建与位移分析系统

## 项目简介
本项目实现了基于双目立体视觉的三维重建与位移分析，支持超分辨率（Super-Resolution, SR）和普通分辨率两种完整流程。主要功能包括：
- 相机标定与立体校正
- 超分辨率图像批量处理（EDSR模型）
- 视差计算与参数搜索
- 特征点匹配与三维重建
- 位移统计与可视化
- 支持一键切换超分/非超分全流程

## 目录结构
- `main.py`：主流程入口，自动串联所有步骤
- `stereo_calib.py`：相机标定与立体校正
- `EDSR/`：超分辨率模型与推理脚本
- `disparity_compute.py`：视差计算
- `feature_match.py`：特征点匹配
- `reproject_3d.py`：三维重建与位移分析
- `plot_displacement_stats.py`：位移统计与可视化
- `output/`：普通分辨率流程输出
- `output_sr/`：超分辨率流程输出

## 依赖环境
建议使用 Python 3.10+，推荐使用虚拟环境（venv）。

### 主要依赖包
- opencv-python
- opencv-contrib-python
- numpy
- matplotlib
- torch
- pillow
- scikit-image
- tqdm

### 安装依赖
1. 确保已安装 Python 3.10 及以上版本。
2. 在项目根目录下，执行以下命令安装所有依赖包：

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib torch pillow scikit-image tqdm
```

---

## 使用说明

### 1. 超分辨率批量处理
#### 标定板图片超分
- 运行以下命令，对标定板图片批量超分，输出到 `../textL18_SR/` 和 `../textR18/textL18_SR/`：
```bash
python3 EDSR/inference_stereo.py
```

#### 散斑图片超分
- 运行以下命令，对散斑图片批量超分，输出到 `../capture_SR/Left/` 和 `../capture_SR/Right/`：
- 如需处理其他目录图片，可在 `EDSR/inference.py` 中修改 `lr_folder` 和 `sr_folder` 路径。
```bash
python3 EDSR/inference.py
```

### 2. 主流程一键运行
- 普通分辨率流程：
```bash
python3 main.py
```
- 超分辨率流程（自动切换所有输入输出到SR目录）：
```bash
python3 main.py --superres
```

### 3. 查看输出结果
- 普通流程输出在 `output/` 及其子目录
- 超分流程输出在 `output_sr/` 及其子目录
- 标定参数、视差图、特征点、三维点、统计图等均自动区分存放

---

## 常见问题
- 若遇到 `cv2`、`torch` 等导入错误，请确认已安装所有依赖。
- 若输出目录无结果，请检查输入图片路径和参数设置，或查看终端报错信息。

## 其他说明
- 所有脚本均支持 `--superres` 参数，自动切换超分/非超分流程。
- 如需自定义输入输出路径，请在对应脚本顶部修改。

## 路径自定义说明
如需自定义输入、输出路径，请在对应脚本顶部查找和修改相关变量。例如：

- **EDSR/inference.py**
  - 修改 `lr_folder` 和 `sr_folder` 变量，分别指定输入和输出图片文件夹。
- **EDSR/inference_stereo.py**
  - 修改 `superres_batch` 函数调用中的路径参数。
- **stereo_calib.py**
  - 修改 `images_left`、`images_right`、`output_dir` 变量。
- **disparity_compute.py**
  - 修改 `left_list`、`right_list`、`output_dir` 变量。
- **feature_match.py**
  - 修改 `left_list`、`disp_dir`、`output_dir` 变量。
- **reproject_3d.py**
  - 修改 `disp_dir`、`match_dir`、`output_dir` 变量。

---

## 2D DIC 位移分析自动化流程

### 目录结构（2D部分）
- `main.py`：2D主流程入口，自动串联标定、校正、视差计算、位移分析、可视化等步骤
- `optimize.py`：自动参数调优脚本，根据理论物理位移自动搜索最优视差参数
- `config.py`：集中管理所有参数，包括数据路径、标定参数、视差参数、理论位移等
- `DIC_2D/`：2D功能模块目录
  - `calibration/`：标定与立体校正
  - `disparity/`：视差计算（SGBM/ZNCC）
  - `analysis/`：位移分析与可视化

### 依赖环境
同上，需安装 `opencv-python`、`numpy`、`matplotlib`、`scipy` 等。

### 2D流程一键运行
```bash
python3 main.py
```
- 自动完成标定、校正、视差计算、位移分析、结果可视化。
- 所有参数自动读取 `config.py`，无需命令行交互。

### 2D参数自动调优
- 支持自动遍历 SGBM 参数（如 blockSize、numDisparities），以**物理位移结果最接近理论值**为目标自动调优。
- 理论位移值可在 `config.py` 中设置：
  ```python
  THEORETICAL_DISPLACEMENT_MM = 1.0  # 单位mm
  ```
- 若未找到最优参数文件，`main.py` 会自动调用 `optimize.py` 完成调优，结果保存在 `output2D/optimize/optimal_params.json`。

### 主要输出
- `output2D/calibration/`：标定参数与校正结果
- `output2D/disparity/`：视差图
- `output2D/analysis/`：位移统计、物理位移（mm）、像素位移等
- `output2D/visualization/`：位移曲线、直方图等可视化结果
- `output2D/optimize/optimal_params.json`：自动调优得到的最优参数及误差

### 主要参数说明（config.py）
- 数据路径：`LEFT_DIR`, `RIGHT_DIR`, `CALIB_LEFT_DIR`, `CALIB_RIGHT_DIR`
- 棋盘格参数：`CHECKERBOARD`, `SQUARE_SIZE`
- 视差方法与参数：`DISPARITY_METHOD`, `SGBM_PARAMS`, `ZNCC_PARAMS`
- 有效视差区间：`DISP_VALID_LOW`, `DISP_VALID_HIGH`
- 异常值剔除：`OUTLIER_METHOD`, `OUTLIER_THRESHOLD`
- **理论物理位移**：`THEORETICAL_DISPLACEMENT_MM`（自动调优目标）

### 典型流程
1. 配置好 `config.py` 路径和参数，设置理论位移值。
2. 运行 `python3 main.py`，系统自动完成所有步骤。
3. 查看 `output2D/` 下的各类结果和最优参数。

---

如有问题请联系开发者或查阅代码注释。 