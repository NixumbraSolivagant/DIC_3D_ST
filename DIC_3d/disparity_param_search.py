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
import optuna

param_dir = 'output_sr' if use_superres else 'output'
params = np.load(os.path.join(param_dir, 'calib_params.npz'))
mapLx, mapLy = params['mapLx'], params['mapLy']
mapRx, mapRy = params['mapRx'], params['mapRy']

left_list = sorted(glob.glob('../capture/Left/*.png'))
right_list = sorted(glob.glob('../capture/Right/*.png'))
# 只用一组图片做参数优化
imgL = cv2.imread(left_list[0])
imgR = cv2.imread(right_list[0])
rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
grayL = cv2.GaussianBlur(grayL, (5,5), 0)
grayR = cv2.GaussianBlur(grayR, (5,5), 0)

def objective(trial):
    numDisparities = trial.suggest_int('numDisparities', 64, 192, step=16)
    blockSize = trial.suggest_int('blockSize', 5, 15, step=2)
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
        stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16
    disp = np.clip(disp, 0, numDisparities)
    valid = np.logical_and(disp > 1, disp < numDisparities-8)
    valid_ratio = np.sum(valid)/valid.size
    return valid_ratio

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)  # 可调整n_trials

best_params = study.best_params
print(f'贝叶斯优化最佳参数: {best_params}')

# 保存最佳参数
param_save_path = os.path.join(param_dir, 'best_disparity_params.json')
with open(param_save_path, 'w') as f:
    json.dump(best_params, f)
print(f'最佳参数已保存到 {param_save_path}') 