import cv2
import numpy as np
import glob
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
# 修正matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import os

# ========== 1. 相机标定和立体校正 ==========
CHECKERBOARD = (11, 8)
square_size = 5  # mm
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size
objpoints = []
imgpoints_left = []
imgpoints_right = []
images_left = sorted(glob.glob('/home/herzog/DIC/textL18/*.png'))
images_right = sorted(glob.glob('/home/herzog/DIC/textR18/textL18/*.png'))
for fname_left, fname_right in zip(images_left, images_right):
    imgL = cv2.imread(fname_left)
    imgR = cv2.imread(fname_right)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
    if retL and retR:
        objpoints.append(objp)
        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)
flags = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, grayL.shape[::-1],
    criteria=criteria, flags=flags
)
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=1
)
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, grayR.shape[::-1], cv2.CV_32FC1)
print("相机标定与立体校正完成。")
# ====== 标定参数保存 ======
np.savez('calib_params.npz', mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, Q=Q, mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy, roi1=roi1, roi2=roi2)
with open('calib_params.txt', 'w') as f:
    def p(s):
        print(s)
        f.write(s+'\n')
    p('===== 标定与立体校正参数 =====')
    p(f'mtxL (左相机内参):\n{mtxL}')
    p(f'distL (左相机畸变): {distL.ravel()}')
    p(f'mtxR (右相机内参):\n{mtxR}')
    p(f'distR (右相机畸变): {distR.ravel()}')
    p(f'R (旋转):\n{R}')
    p(f'T (平移): {T.ravel()}')
    p(f'Q (重投影矩阵):\n{Q}')
    p(f'roi1: {roi1}, roi2: {roi2}')
    p('===========================')
# ====== 标定结果图片保存 ======
output_dir = 'calib_output'
os.makedirs(output_dir, exist_ok=True)
if len(images_left) > 0 and len(images_right) > 0:
    imgL = cv2.imread(images_left[0])
    imgR = cv2.imread(images_right[0])
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    rectL_cropped = rectL[y1:y1+h1, x1:x1+w1]
    rectR_cropped = rectR[y2:y2+h2, x2:x2+w2]
    cv2.imwrite(os.path.join(output_dir, 'imgL_before.png'), imgL)
    cv2.imwrite(os.path.join(output_dir, 'imgR_before.png'), imgR)
    cv2.imwrite(os.path.join(output_dir, 'imgL_after.png'), rectL_cropped)
    cv2.imwrite(os.path.join(output_dir, 'imgR_after.png'), rectR_cropped)
    h1, w1 = rectL_cropped.shape[:2]
    h2, w2 = rectR_cropped.shape[:2]
    h = min(h1, h2)
    rectL_cropped = rectL_cropped[:h, :]
    rectR_cropped = rectR_cropped[:h, :]
    concat_img = np.hstack((rectL_cropped, rectR_cropped))
    interval = 20
    for y in range(interval, h, interval):
        cv2.line(concat_img, (0, y), (w1 + w2, y), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(output_dir, 'rectified_concat.png'), concat_img)
    imgL_vis = imgL.copy()
    imgR_vis = imgR.copy()
    grayL_vis = cv2.cvtColor(imgL_vis, cv2.COLOR_BGR2GRAY)
    grayR_vis = cv2.cvtColor(imgR_vis, cv2.COLOR_BGR2GRAY)
    retL_vis, cornersL_vis = cv2.findChessboardCorners(grayL_vis, CHECKERBOARD, None)
    retR_vis, cornersR_vis = cv2.findChessboardCorners(grayR_vis, CHECKERBOARD, None)
    if retL_vis:
        cv2.drawChessboardCorners(imgL_vis, CHECKERBOARD, cornersL_vis, retL_vis)
    if retR_vis:
        cv2.drawChessboardCorners(imgR_vis, CHECKERBOARD, cornersR_vis, retR_vis)
    cv2.imwrite(os.path.join(output_dir, 'cornersL.png'), imgL_vis)
    cv2.imwrite(os.path.join(output_dir, 'cornersR.png'), imgR_vis)
    cornersL_rect = cv2.undistortPoints(imgpoints_left[0], mtxL, distL, R=R1, P=P1).reshape(-1,2)
    cornersR_rect = cv2.undistortPoints(imgpoints_right[0], mtxR, distR, R=R2, P=P2).reshape(-1,2)
    yL = cornersL_rect[:,1]
    yR = cornersR_rect[:,1]
    plt.figure(figsize=(8,4))
    plt.plot(yL, 'r.-', label='左图校正后角点y')
    plt.plot(yR, 'b.-', label='右图校正后角点y')
    plt.plot(np.abs(yL-yR), 'g.-', label='y坐标差')
    plt.legend()
    plt.title('校正后极线对齐误差（y坐标）')
    plt.xlabel('角点序号')
    plt.ylabel('像素')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rectified_y_error.png'))
    plt.close()
rms_errors = []
for i in range(len(objpoints)):
    imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
    imgpoints_proj = imgpoints_proj.reshape(-1, 2)
    imgpoints_detected = imgpoints_left[i].reshape(-1, 2)
    error = np.linalg.norm(imgpoints_proj - imgpoints_detected, axis=1)
    rms = np.sqrt(np.mean(error**2))
    rms_errors.append(rms)
plt.figure(figsize=(8, 4))
plt.plot(rms_errors, 'o-', label='重投影RMS误差')
plt.xlabel('图片编号')
plt.ylabel('重投影RMS误差 (像素)')
plt.title('每张标定图片的重投影RMS误差')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reproj_rms_error.png'))
plt.close()