import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import config

def run_rectification(calib_data, cfg):
    output_dir = cfg.CALIB_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    mtxL = calib_data['mtxL']
    distL = calib_data['distL']
    mtxR = calib_data['mtxR']
    distR = calib_data['distR']
    objpoints = calib_data['objpoints']
    imgpoints_left = calib_data['imgpoints_left']
    imgpoints_right = calib_data['imgpoints_right']
    image_size = calib_data['grayL_shape']
    rvecsL = calib_data['rvecsL']
    tvecsL = calib_data['tvecsL']
    # stereoCalibrate
    flags = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtxL, distL, mtxR, distR, image_size,
        criteria=criteria, flags=flags
    )
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL, mtxR, distR, image_size, R, T, alpha=1
    )
    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)
    # 保存参数
    np.savez(os.path.join(output_dir, 'rectify_params.npz'), Q=Q, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, roi1=roi1, roi2=roi2, mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy)
    # Q矩阵与基线检验
    with open(os.path.join(output_dir, 'Qmatrix_baseline_check.txt'), 'w') as f:
        f.write(f'Q matrix:\n{Q}\n')
        if Q.shape == (4, 4):
            f_q = Q[2, 3]
            inv_baseline = Q[3, 2]
            cx_q = -Q[0, 3]
            cy_q = -Q[1, 3]
            if inv_baseline != 0:
                baseline_q = -1.0 / inv_baseline
            else:
                baseline_q = float('inf')
            f.write(f'Q[2,3] (f): {f_q:.4f}\nQ[3,2] (-1/B): {inv_baseline:.6f}\nQ主点(cx, cy): ({cx_q:.2f}, {cy_q:.2f})\nQ矩阵推算基线B: {baseline_q:.4f}\n')
            if inv_baseline >= 0:
                f.write('[警告] Q[3,2] 应为负数，基线B应为正，当前Q[3,2]为正，可能有问题！\n')
            if not (10 < baseline_q < 1000):
                f.write('[警告] 基线长度B异常，通常应在10~1000单位之间，请检查标定参数！\n')
        else:
            f.write('[警告] Q矩阵不是4x4，无法自动检验！\n')
        baseline_T = np.linalg.norm(T)
        f.write(f'T向量推算基线B: {baseline_T:.4f}\n')
        if 'baseline_q' in locals() and abs(baseline_T - baseline_q) > 1e-2:
            f.write(f'[警告] Q矩阵与T向量推算的基线不一致，差值: {abs(baseline_T - baseline_q):.4f}\n')
        f.write('===== Q矩阵与基线参数检验结束 =====\n')
    # 极线误差分析
    def compute_epipolar_error(F, pts1, pts2):
        pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
        pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
        l2 = (F @ pts1_h.T).T
        l1 = (F.T @ pts2_h.T).T
        d1 = np.abs(np.sum(l2 * pts2_h, axis=1)) / np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2)
        d2 = np.abs(np.sum(l1 * pts1_h, axis=1)) / np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)
        return d1, d2
    pts1 = np.vstack([c.reshape(-1, 2) for c in imgpoints_left])
    pts2 = np.vstack([c.reshape(-1, 2) for c in imgpoints_right])
    d1, d2 = compute_epipolar_error(F, pts1, pts2)
    np.savetxt(os.path.join(output_dir, 'epipolar_error_left2right.txt'), d1, fmt='%.6f')
    np.savetxt(os.path.join(output_dir, 'epipolar_error_right2left.txt'), d2, fmt='%.6f')
    with open(os.path.join(output_dir, 'epipolar_error_summary.txt'), 'w') as f:
        f.write(f'极线几何误差均值(左到右): {np.mean(d1):.4f}\n')
        f.write(f'极线几何误差均值(右到左): {np.mean(d2):.4f}\n')
        f.write(f'极线几何误差最大值(左到右): {np.max(d1):.4f}\n')
        f.write(f'极线几何误差最大值(右到左): {np.max(d2):.4f}\n')
    # 校正后左右角点y坐标差
    cornersL_rect = cv2.undistortPoints(imgpoints_left[0], mtxL, distL, R=R1, P=P1).reshape(-1, 2)
    cornersR_rect = cv2.undistortPoints(imgpoints_right[0], mtxR, distR, R=R2, P=P2).reshape(-1, 2)
    y_diff = np.abs(cornersL_rect[:, 1] - cornersR_rect[:, 1])
    mean_y_diff = np.mean(y_diff)
    max_y_diff = np.max(y_diff)
    np.savetxt(os.path.join(output_dir, 'rectified_y_diff.txt'), y_diff, fmt='%.6f')
    with open(os.path.join(output_dir, 'rectified_y_diff_summary.txt'), 'w') as f:
        f.write(f'校正后左右角点y坐标差均值: {mean_y_diff:.4f}\n最大值: {max_y_diff:.4f}\n')
    # 校正前后图像与极线拼接
    imgL = cv2.imread(sorted(glob.glob(os.path.join(cfg.CALIB_LEFT_DIR, '*.png')))[0])
    imgR = cv2.imread(sorted(glob.glob(os.path.join(cfg.CALIB_RIGHT_DIR, '*.png')))[0])
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
    x1, y1, w1, h1 = roi1 = [int(x) for x in roi1]
    x2, y2, w2, h2 = roi2 = [int(x) for x in roi2]
    rectL_cropped = rectL[y1:y1 + h1, x1:x1 + w1]
    rectR_cropped = rectR[y2:y2 + h2, x2:x2 + w2]
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    plt.title('校正前左图')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
    plt.title('校正前右图')
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(rectL_cropped, cv2.COLOR_BGR2RGB))
    plt.title('校正后左图(有效区域)')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(rectR_cropped, cv2.COLOR_BGR2RGB))
    plt.title('校正后右图(有效区域)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_rectified_images.png'))
    plt.close()
    # 极线拼接
    h1, w1 = rectL_cropped.shape[:2]
    h2, w2 = rectR_cropped.shape[:2]
    h = min(h1, h2)
    rectL_cropped = rectL_cropped[:h, :]
    rectR_cropped = rectR_cropped[:h, :]
    concat_img = np.hstack((rectL_cropped, rectR_cropped))
    interval = 20
    for y in range(interval, h, interval):
        cv2.line(concat_img, (0, y), (w1 + w2, y), (0, 0, 255), 1)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(concat_img, cv2.COLOR_BGR2RGB))
    plt.title('校正后左右图像拼合及极线')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'rectified_images_concat.png'))
    plt.close()
    # 保存ROI参数
    np.savez(os.path.join(output_dir, 'roi_params.npz'), roi1=roi1, roi2=roi2)
    return dict(Q=Q, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, roi1=roi1, roi2=roi2, mapLx=mapLx, mapLy=mapLy, mapRx=mapRx, mapRy=mapRy, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR) 