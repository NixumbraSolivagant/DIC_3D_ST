import sys
use_superres = False
if '--superres' in sys.argv:
    use_superres = True
import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import json

param_dir = 'output_sr' if use_superres else 'output'
params = np.load(os.path.join(param_dir, 'calib_params.npz'))
mapLx, mapLy = params['mapLx'], params['mapLy']
mapRx, mapRy = params['mapRx'], params['mapRy']
Q = params['Q']

# 读取最佳参数
param_file = os.path.join(param_dir, 'best_disparity_params.json')
if os.path.exists(param_file):
    with open(param_file, 'r') as f:
        best_params = json.load(f)
    numDisparities = best_params.get('numDisparities', 64)
    blockSize = best_params.get('blockSize', 9)
    print(f'使用自动优化参数: numDisparities={numDisparities}, blockSize={blockSize}')
else:
    numDisparities = 64
    blockSize = 9
    print('未找到最佳参数文件，使用默认参数。')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

if use_superres:
    left_list = sorted(glob.glob('../capture_SR/Left/*.png'))
    right_list = sorted(glob.glob('../capture_SR/Right/*.png'))
    output_dir = 'output_sr/output_disp'
    ensure_dir('output_sr')
else:
    left_list = sorted(glob.glob('../capture/Left/*.png'))
    right_list = sorted(glob.glob('../capture/Right/*.png'))
    output_dir = 'output/output_disp'
    ensure_dir('output')
ensure_dir(output_dir)
if hasattr(cv2, 'StereoSGBM_create'):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisparities,
        blockSize=blockSize,
        P1=8*3*blockSize**2,
        P2=32*3*blockSize**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=16
    )
else:
    print('警告：未检测到StereoSGBM，自动降级为StereoBM，建议安装opencv-contrib-python以获得更好视差效果。')
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
for idx, (lf, rf) in enumerate(zip(left_list, right_list)):
    imgL = cv2.imread(lf)
    imgR = cv2.imread(rf)
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    grayL = cv2.GaussianBlur(grayL, (5,5), 0)
    grayR = cv2.GaussianBlur(grayR, (5,5), 0)
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16
    disp = cv2.medianBlur(disp, 5)
    disp = np.clip(disp, 0, 128)
    np.save(os.path.join(output_dir, f'disp_{idx+1}.npy'), disp)
    plt.figure(figsize=(6,4))
    plt.imshow(disp, cmap='jet')
    plt.colorbar()
    plt.title(f'disp_{idx+1}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'disp_{idx+1}.png'))
    plt.close()
    plt.figure()
    plt.hist(disp.ravel(), bins=50, range=(0, 128))
    plt.title(f'disp_{idx+1} histogram')
    plt.xlabel('Disparity')
    plt.ylabel('Pixel count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'disp_{idx+1}_hist.png'))
    plt.close()
    valid = np.logical_and(disp > 1, disp < 120)
    print(f'disp_{idx+1}: 有效视差像素比例 {np.sum(valid)/valid.size:.4f}')
print('所有视差图已保存到 output_disp/') 