import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import config
import gc

def compute_disparity(rectify_data, cfg):
    output_dir = cfg.DISPARITY_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    mapLx = rectify_data['mapLx']
    mapLy = rectify_data['mapLy']
    mapRx = rectify_data['mapRx']
    mapRy = rectify_data['mapRy']
    roi1 = rectify_data['roi1']
    roi2 = rectify_data['roi2']
    left_list = sorted([os.path.join(cfg.LEFT_DIR, f) for f in os.listdir(cfg.LEFT_DIR) if f.endswith('.png')])
    right_list = sorted([os.path.join(cfg.RIGHT_DIR, f) for f in os.listdir(cfg.RIGHT_DIR) if f.endswith('.png')])
    disparities = []
    for idx, (left_path, right_path) in enumerate(zip(left_list, right_list)):
        try:
            imgL = cv2.imread(left_path)
            imgR = cv2.imread(right_path)
            rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
            rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
            # ROI裁剪
            x1, y1, w1, h1 = [int(x) for x in roi1]
            x2, y2, w2, h2 = [int(x) for x in roi2]
            rectL_roi = rectL[y1:y1+h1, x1:x1+w1]
            rectR_roi = rectR[y2:y2+h2, x2:x2+w2]
            # 自动修复：调整为灰度、uint8，并裁剪为相同尺寸
            grayL = cv2.cvtColor(rectL_roi, cv2.COLOR_BGR2GRAY) if rectL_roi.ndim == 3 else rectL_roi
            grayR = cv2.cvtColor(rectR_roi, cv2.COLOR_BGR2GRAY) if rectR_roi.ndim == 3 else rectR_roi
            grayL = grayL.astype(np.uint8)
            grayR = grayR.astype(np.uint8)
            # 自动修复：强制裁剪为相同尺寸
            min_h = min(grayL.shape[0], grayR.shape[0])
            min_w = min(grayL.shape[1], grayR.shape[1])
            grayL = grayL[:min_h, :min_w]
            grayR = grayR[:min_h, :min_w]
            if grayL.shape != grayR.shape or grayL.dtype != np.uint8 or grayR.dtype != np.uint8:
                raise ValueError(f"输入SGBM的左右图像尺寸或类型不一致！\n"
                                 f"grayL.shape: {grayL.shape}, grayR.shape: {grayR.shape}\n"
                                 f"grayL.dtype: {grayL.dtype}, grayR.dtype: {grayR.dtype}")
            cv2.imwrite(os.path.join(output_dir, f'group{idx+1}_L.png'), grayL)
            cv2.imwrite(os.path.join(output_dir, f'group{idx+1}_R.png'), grayR)
            sgbm = cv2.StereoSGBM_create(**cfg.SGBM_PARAMS)
            disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
            np.save(os.path.join(output_dir, f'disp_{idx+1}.npy'), disp)
            # 可视化视差图
            valid_disp = disp[(disp > 0) & (disp < cfg.SGBM_PARAMS['numDisparities'])]
            vmax = np.percentile(valid_disp, 98) if valid_disp.size > 0 else cfg.SGBM_PARAMS['numDisparities']
            plt.figure(figsize=(6, 5))
            plt.imshow(disp, cmap='jet', vmin=0, vmax=vmax)
            plt.title(f'SGBM视差图-组{idx+1}')
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'disp_{idx+1}.png'))
            plt.close()
            disparities.append(disp)
        except Exception as e:
            print(f"[警告] 处理第{idx+1}组图片时出错: {e}")
        finally:
            del imgL, imgR, rectL, rectR, rectL_roi, rectR_roi, grayL, grayR
            gc.collect()
    # 棋盘格标定图像视差图可视化
    # 取第一组标定图像
    calib_left = sorted([os.path.join(cfg.CALIB_LEFT_DIR, f) for f in os.listdir(cfg.CALIB_LEFT_DIR) if f.endswith('.png')])
    calib_right = sorted([os.path.join(cfg.CALIB_RIGHT_DIR, f) for f in os.listdir(cfg.CALIB_RIGHT_DIR) if f.endswith('.png')])
    if calib_left and calib_right:
        imgL = cv2.imread(calib_left[0], cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(calib_right[0], cv2.IMREAD_GRAYSCALE)
        rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)
        sgbm = cv2.StereoSGBM_create(**cfg.SGBM_PARAMS)
        disp = sgbm.compute(rectL, rectR).astype(np.float32) / 16.0
        valid_disp = disp[disp > 0]
        vmax = np.percentile(valid_disp, 98) if valid_disp.size > 0 else cfg.SGBM_PARAMS['numDisparities']
        plt.figure(figsize=(6, 5))
        plt.imshow(disp, cmap='jet', vmin=0, vmax=vmax)
        plt.title('棋盘格标定图像视差图 (SGBM)')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'chessboard_disparity_map.png'))
        plt.close()
    return disparities 