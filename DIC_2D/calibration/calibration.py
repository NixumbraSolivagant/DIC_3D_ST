import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import config

def run_calibration(cfg):
    CHECKERBOARD = cfg.CHECKERBOARD
    square_size = cfg.SQUARE_SIZE
    output_dir = cfg.CALIB_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    images_left = sorted(glob.glob(os.path.join(cfg.CALIB_LEFT_DIR, '*.png')))
    images_right = sorted(glob.glob(os.path.join(cfg.CALIB_RIGHT_DIR, '*.png')))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    for i, (fname_left, fname_right) in enumerate(zip(images_left, images_right)):
        imgL = cv2.imread(fname_left)
        imgR = cv2.imread(fname_right)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if retL:
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        if retR:
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        if retL and retR:
            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)
        # 可视化角点检测
        visL = imgL.copy()
        visR = imgR.copy()
        if retL:
            cv2.drawChessboardCorners(visL, CHECKERBOARD, cornersL, retL)
        if retR:
            cv2.drawChessboardCorners(visR, CHECKERBOARD, cornersR, retR)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(visL, cv2.COLOR_BGR2RGB))
        plt.title(f'左图角点检测 {i+1}')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(visR, cv2.COLOR_BGR2RGB))
        plt.title(f'右图角点检测 {i+1}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'chessboard_corners_{i+1}.png'))
        plt.close()
    # 标定
    init_camera = np.eye(3, dtype=np.float64)
    init_dist = np.zeros((5, 1), dtype=np.float64)
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], init_camera.copy(), init_dist.copy())
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], init_camera.copy(), init_dist.copy())
    # 保存参数
    np.savez(os.path.join(output_dir, 'calib_params.npz'), mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, objpoints=objpoints, imgpoints_left=imgpoints_left, imgpoints_right=imgpoints_right, rvecsL=rvecsL, tvecsL=tvecsL)
    # 重投影误差分析
    rms_errors = []
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecsL[i], tvecsL[i], mtxL, distL)
        imgpoints_proj = imgpoints_proj.reshape(-1, 2)
        imgpoints_detected = imgpoints_left[i].reshape(-1, 2)
        error = np.linalg.norm(imgpoints_proj - imgpoints_detected, axis=1)
        rms = np.sqrt(np.mean(error ** 2))
        rms_errors.append(rms)
    np.savetxt(os.path.join(output_dir, 'reprojection_rms_errors.txt'), rms_errors, fmt='%.6f')
    plt.figure(figsize=(8, 4))
    plt.plot(rms_errors, 'o-', label='重投影RMS误差')
    plt.xlabel('图片编号')
    plt.ylabel('重投影RMS误差 (像素)')
    plt.title('每张标定图片的重投影RMS误差')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reprojection_rms_errors.png'))
    plt.close()
    # 像素物理尺寸
    pixel_size_mm = None
    if len(imgpoints_left) > 0:
        pts = imgpoints_left[0].reshape(-1, 2)
        row0 = pts[:CHECKERBOARD[0], 0]
        px_per_square = np.abs(np.mean(np.diff(row0)))
        pixel_size_mm = square_size / px_per_square
        pixel_size_um = pixel_size_mm * 1000
        with open(os.path.join(output_dir, 'pixel_size.txt'), 'w') as f:
            f.write(f'pixel_size_mm: {pixel_size_mm:.6f}\npixel_size_um: {pixel_size_um:.2f}\n')
    else:
        pixel_size_mm = None
    return dict(mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, objpoints=objpoints, imgpoints_left=imgpoints_left, imgpoints_right=imgpoints_right, grayL_shape=grayL.shape[::-1], rvecsL=rvecsL, tvecsL=tvecsL, pixel_size_mm=pixel_size_mm) 