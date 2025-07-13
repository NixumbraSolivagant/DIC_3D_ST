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

# 安装依赖
pip install opencv-python opencv-contrib-python numpy matplotlib torch pillow scikit-image tqdm
```

## 超分辨率批量处理
### 1. 标定板图片超分
运行 `EDSR/inference_stereo.py`，对标定板图片批量超分，输出到 `../textL18_SR/` 和 `../textR18/textL18_SR/`。
```bash
python3 EDSR/inference_stereo.py
```

### 2. 散斑图片超分
运行 `EDSR/inference.py`，对散斑图片批量超分，输出到 `../capture_SR/Left/` 和 `../capture_SR/Right/`。
- 你可以在 `EDSR/inference.py` 中修改 `lr_folder` 和 `sr_folder` 路径，分别处理左/右相机图片。
```bash
python3 EDSR/inference.py
```

## 主流程一键运行
激活虚拟环境后，运行：
```bash
# 普通分辨率流程
python3 main.py

# 超分辨率流程（自动切换所有输入输出到SR目录）
python3 main.py --superres
```

## 输出目录说明
- 普通流程输出：`output/` 及其子目录
- 超分流程输出：`output_sr/` 及其子目录
- 标定参数、视差图、特征点、三维点、统计图等均自动区分存放

## 常见问题
- 若遇到 `cv2`、`torch` 等导入错误，请确认已激活虚拟环境并安装依赖。
- 若输出目录无结果，请检查输入图片路径和参数设置，或查看终端报错信息。

## 其他说明
- 所有脚本均支持 `--superres` 参数，自动切换超分/非超分流程。
- 如需自定义输入输出路径，请在对应脚本顶部修改。

---
如有问题请联系开发者或查阅代码注释。 