import os

# 数据目录
DATA_ROOT = '../capture0.1'
LEFT_DIR = os.path.join(DATA_ROOT, 'Left')
RIGHT_DIR = os.path.join(DATA_ROOT, 'Right')
CALIB_LEFT_DIR = '../text0.1/Right'
CALIB_RIGHT_DIR = '../text0.1/Left'

# 输出目录
OUTPUT_ROOT = 'output2D_0.1'
CALIB_OUTPUT = os.path.join(OUTPUT_ROOT, 'calibration')
DISPARITY_OUTPUT = os.path.join(OUTPUT_ROOT, 'disparity')
ANALYSIS_OUTPUT = os.path.join(OUTPUT_ROOT, 'analysis')
VISUAL_OUTPUT = os.path.join(OUTPUT_ROOT, 'visualization')

# 棋盘格参数
CHECKERBOARD = (11, 8)
SQUARE_SIZE = 5  # mm

# 视差方法选择
DISPARITY_METHOD = 'SGBM'  # 可选 'SGBM', 'BM', 'ZNCC'

# SGBM参数
SGBM_PARAMS = {
    'minDisparity': 0,
    'numDisparities': 128,
    'blockSize': 5,
    'P1': 8 * 1 * 5 ** 2,
    'P2': 32 * 1 * 5 ** 2,
    'disp12MaxDiff': 1,
    'uniquenessRatio': 5,
    'speckleWindowSize': 50,
    'speckleRange': 16,
    'mode': 3  # cv2.STEREO_SGBM_MODE_SGBM_3WAY
}

# ZNCC参数
ZNCC_PARAMS = {
    'max_disp': 64,
    'win_size': 7,
    'step': 2,
    'zncc_thresh': 0.1
}

# 有效视差区间
DISP_VALID_LOW = 1
DISP_VALID_HIGH = 60

# 异常值剔除方法
OUTLIER_METHOD = 'zscore'  # 可选: 'iqr', 'zscore', 'mad', 'percentile'
OUTLIER_THRESHOLD = 3 
# 理论物理位移（mm），用于自动调参目标
THEORETICAL_DISPLACEMENT_MM = 0.1