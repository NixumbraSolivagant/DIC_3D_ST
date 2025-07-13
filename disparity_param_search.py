import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

params = np.load('output/calib_params.npz')
mapLx, mapLy = params['mapLx'], params['mapLy']
mapRx, mapRy = params['mapRx'], params['mapRy']

left_list = sorted(glob.glob('../capture/Left/*.png'))
right_list = sorted(glob.glob('../capture/Right/*.png'))
imgL = cv2.imread(left_list[0])
imgR = cv2.imread(right_list[0])
rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
grayL = cv2.GaussianBlur(grayL, (5,5), 0)
grayR = cv2.GaussianBlur(grayR, (5,5), 0)

numDisparities_list = [64, 96, 128, 160, 192]
blockSize_list = [5, 7, 9, 11, 13]

results = []
for numDisp in numDisparities_list:
    for blk in blockSize_list:
        if hasattr(cv2, 'StereoSGBM_create'):
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=numDisp,
                blockSize=blk,
                P1=8*3*blk**2,
                P2=32*3*blk**2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=16
            )
        else:
            stereo = cv2.StereoBM_create(numDisparities=numDisp, blockSize=blk)
        disp = stereo.compute(grayL, grayR).astype(np.float32) / 16
        disp = np.clip(disp, 0, numDisp)
        valid = np.logical_and(disp > 1, disp < numDisp-8)
        valid_ratio = np.sum(valid)/valid.size
        results.append((numDisp, blk, valid_ratio))
        print(f'numDisparities={numDisp}, blockSize={blk}: 有效视差比例={valid_ratio:.4f}')

# 输出最佳参数
best = max(results, key=lambda x: x[2])
print(f'最佳参数: numDisparities={best[0]}, blockSize={best[1]}, 有效视差比例={best[2]:.4f}') 